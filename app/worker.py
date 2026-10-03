from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy import func, or_, select, update

from app.config import Settings
from app.curation import generate_collection_specs, persist_collection_specs
from app.database import session_scope
from app.image_analysis import ImageAnalyzer, ImageTooLargeError, PlaceResolver
from app.models import AppState, Face, Job, Photo, utcnow
from app.source import PhotoSource, SourceEntry, SourceSizeLimitError, build_source

logger = logging.getLogger(__name__)


class JobCancelled(RuntimeError):
    pass


class WorkerStopping(RuntimeError):
    pass


class WorkerService:
    """Single durable background worker.

    Jobs and per-photo analysis state live in SQLite, so a container restart can
    safely resume. Source photos are only ever accessed through the read-only
    PhotoSource interface.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._analyzer: ImageAnalyzer | None = None
        self._place_resolver = PlaceResolver()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.clear()
        self._recover_interrupted_work()
        self._cleanup_stale_temp_files()
        if self.settings.auto_start:
            self._ensure_initial_job()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="photo-curator-worker",
            daemon=True,
        )
        self._thread.start()

    def _recover_interrupted_work(self) -> None:
        with session_scope() as session:
            interrupted = list(session.scalars(select(Job).where(Job.status == "running")))
            for job in interrupted:
                job.status = "queued"
                # Preserve the durable phase checkpoint. A full job that had
                # reached analysis must not walk the whole source again.
                job.message = f"Resuming {job.phase} after application restart"
                job.started_at = None
            session.execute(
                update(Photo)
                .where(Photo.analysis_status == "analyzing")
                .values(
                    analysis_status="error",
                    analysis_error="Analysis was interrupted; it will be retried",
                )
            )

    def _cleanup_stale_temp_files(self) -> None:
        try:
            stale_paths = list(self.settings.temp_dir.iterdir())
            stale_paths.extend(self.settings.thumbs_dir.glob("*/*.tmp"))
            stale_paths.extend(self.settings.previews_dir.glob("*/*.tmp"))
            for path in stale_paths:
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not clean stale analysis files: %s", exc)

    def stop(self, timeout: float = 15) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def enqueue(self, kind: str = "full", force: bool = False) -> int:
        if kind not in {"full", "scan", "analyze", "collections"}:
            raise ValueError(f"Unknown job kind: {kind}")
        with session_scope() as session:
            if not force:
                active_kinds = ("full",) if kind == "full" else (kind, "full")
                active = session.scalar(
                    select(Job)
                    .where(
                        Job.status.in_(("queued", "running")),
                        Job.kind.in_(active_kinds),
                    )
                    .order_by(Job.created_at)
                )
                if active:
                    return active.id
            job = Job(kind=kind, status="queued", phase="queued", message="Waiting for worker")
            session.add(job)
            session.flush()
            job_id = job.id
        self._wake_event.set()
        return job_id

    def request_cancel(self, job_id: int) -> bool:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job or job.status not in {"queued", "running"}:
                return False
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.phase = "cancelled"
                job.finished_at = utcnow()
        self._wake_event.set()
        return True

    def _ensure_initial_job(self) -> None:
        with session_scope() as session:
            active = session.scalar(select(func.count()).select_from(Job).where(Job.status.in_(("queued", "running"))))
            analyzed = session.scalar(select(func.count()).select_from(Photo))
        if not active and not analyzed:
            self.enqueue("full")

    def _run_loop(self) -> None:
        logger.info("Background worker started")
        while not self._stop_event.is_set():
            job_id = self._claim_next_job()
            if job_id is None:
                self._schedule_periodic_scan_if_due()
                self._wake_event.wait(self.settings.worker_poll_seconds)
                self._wake_event.clear()
                continue
            try:
                self._execute_job(job_id)
            except WorkerStopping:
                self._requeue_job(job_id, "Paused for application shutdown; will resume")
            except JobCancelled:
                self._finish_job(job_id, "cancelled", "Cancelled by user")
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                self._finish_job(job_id, "failed", "Job failed", str(exc))
        logger.info("Background worker stopped")

    def _claim_next_job(self) -> int | None:
        with session_scope() as session:
            job = session.scalar(
                select(Job).where(Job.status == "queued").order_by(Job.created_at, Job.id)
            )
            if not job:
                return None
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = utcnow()
                return None
            job.status = "running"
            if job.phase in {"queued", "starting"}:
                job.phase = "starting"
                job.message = "Starting"
            else:
                job.message = f"Resuming from {job.phase} checkpoint"
            job.started_at = utcnow()
            job.heartbeat_at = utcnow()
            return job.id

    def _execute_job(self, job_id: int) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            kind = job.kind
            resume_phase = job.phase

        run_scan = kind in {"full", "scan"}
        run_analyze = kind in {"full", "analyze"}
        run_collections = kind in {"full", "collections"}
        if kind == "full" and resume_phase == "analyzing":
            run_scan = False
        elif kind == "full" and resume_phase in {"curating", "collections"}:
            run_scan = False
            run_analyze = False

        source: PhotoSource | None = None
        if run_scan or run_analyze:
            connection_phase = "analyzing" if resume_phase == "analyzing" else "connecting"
            self._set_progress(
                job_id,
                connection_phase,
                0,
                0,
                "Reconnecting to read-only source"
                if resume_phase == "analyzing"
                else "Testing read-only source access",
            )
            source = build_source(self.settings)
            source.test_connection()

        if run_scan:
            assert source
            self._scan(job_id, source)
        if run_analyze:
            assert source
            # This phase write is the durable boundary between inventory and
            # expensive per-photo work. Restarts continue from pending/error
            # photo rows and do not repeat the completed source walk.
            self._set_progress(job_id, "analyzing", 0, 0, "Preparing incremental analysis")
            self._analyze(job_id, source)
        if run_collections:
            self._build_collections(job_id)
        self._finish_job(job_id, "completed", "Finished successfully")

    def _scan(self, job_id: int, source: PhotoSource) -> None:
        token = str(uuid.uuid4())
        discovered = 0
        batch: list[SourceEntry] = []
        self._set_progress(job_id, "scanning", 0, 0, "Walking folders")

        def flush(entries: list[SourceEntry]) -> None:
            nonlocal discovered
            if not entries:
                return
            paths = [entry.relative_path for entry in entries]
            with session_scope() as session:
                existing = {
                    photo.relative_path: photo
                    for photo in session.scalars(
                        select(Photo).where(Photo.relative_path.in_(paths))
                    )
                }
                for entry in entries:
                    photo = existing.get(entry.relative_path)
                    if photo is None:
                        photo = Photo(
                            relative_path=entry.relative_path,
                            file_name=entry.name,
                            extension=Path(entry.name).suffix.lower(),
                            size_bytes=entry.size_bytes,
                            source_mtime=entry.mtime,
                            analysis_status="pending",
                            analysis_version=0,
                        )
                        session.add(photo)
                    else:
                        changed = (
                            photo.size_bytes != entry.size_bytes
                            or abs(photo.source_mtime - entry.mtime) > 0.001
                        )
                        photo.file_name = entry.name
                        photo.extension = Path(entry.name).suffix.lower()
                        photo.size_bytes = entry.size_bytes
                        photo.source_mtime = entry.mtime
                        if changed:
                            photo.analysis_status = "pending"
                            photo.analysis_version = 0
                            photo.analysis_attempts = 0
                            photo.analysis_error = None
                    photo.active = True
                    photo.scan_token = token
                discovered += len(entries)
            self._set_progress(
                job_id,
                "scanning",
                discovered,
                0,
                f"Discovered {discovered:,} supported photos",
            )

        scan_limited = False
        for entry in source.walk():
            self._check_cancelled(job_id)
            batch.append(entry)
            if (
                self.settings.scan_max_files
                and discovered + len(batch) >= self.settings.scan_max_files
            ):
                scan_limited = True
                break
            if len(batch) >= max(1, self.settings.worker_batch_size):
                flush(batch)
                batch.clear()
        flush(batch)

        with session_scope() as session:
            if source.walk_warnings == 0 and not scan_limited:
                session.execute(
                    update(Photo)
                    .where(
                        or_(Photo.scan_token.is_(None), Photo.scan_token != token),
                        Photo.active.is_(True),
                    )
                    .values(active=False)
                )
                message = f"Scan complete: {discovered:,} photos indexed"
            elif scan_limited:
                message = (
                    f"QA scan limit reached after {discovered:,} photos; "
                    "missing-file cleanup was skipped for safety"
                )
            else:
                message = (
                    f"Scan indexed {discovered:,} photos with {source.walk_warnings} unreadable "
                    "entries; missing-file cleanup was skipped for safety"
                )
            self._set_state_in_session(session, "last_scan_at", datetime.now(UTC).isoformat())
            self._set_state_in_session(session, "last_scan_count", str(discovered))
            self._set_state_in_session(session, "last_scan_warnings", str(source.walk_warnings))
        self._set_progress(job_id, "scanning", discovered, discovered, message)

    def _analyze(self, job_id: int, source: PhotoSource) -> None:
        if self._analyzer is None:
            self._analyzer = ImageAnalyzer(self.settings)
        max_bytes = self.settings.max_file_mb * 1024 * 1024
        with session_scope() as session:
            photo_ids = list(
                session.scalars(
                    select(Photo.id)
                    .where(
                        Photo.active.is_(True),
                        or_(
                            Photo.analysis_status.in_(("pending", "error", "analyzing")),
                            Photo.analysis_version < self.settings.analysis_version,
                        ),
                        Photo.analysis_attempts < self.settings.max_analysis_attempts,
                    )
                    .order_by(Photo.capture_at.is_(None).desc(), Photo.id)
                )
            )
        total = len(photo_ids)
        self._set_progress(job_id, "analyzing", 0, total, f"{total:,} photos need analysis")

        for index, photo_id in enumerate(photo_ids, 1):
            self._check_cancelled(job_id)
            with session_scope() as session:
                photo = session.get(Photo, photo_id)
                if not photo or not photo.active:
                    continue
                relative_path = photo.relative_path
                extension = photo.extension or ".img"
                source_mtime = photo.source_mtime
                if photo.size_bytes > max_bytes:
                    photo.analysis_status = "too_large"
                    photo.analysis_version = self.settings.analysis_version
                    photo.analysis_error = f"Larger than MAX_FILE_MB={self.settings.max_file_mb}"
                    self._set_job_in_session(
                        session,
                        job_id,
                        "analyzing",
                        index,
                        total,
                        f"Skipped oversized file {index:,}/{total:,}",
                    )
                    continue
                photo.analysis_status = "analyzing"
                # Increment before touching the source so a hard crash on one corrupt
                # or pathological file cannot cause an infinite restart loop.
                photo.analysis_attempts += 1

            cache_key = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
            thumb_relative = f"thumbs/{cache_key[:2]}/{cache_key}.jpg"
            preview_relative = f"previews/{cache_key[:2]}/{cache_key}.jpg"
            thumbnail_path = self.settings.cache_dir / thumb_relative
            preview_path = self.settings.cache_dir / preview_relative
            temporary_path = self.settings.temp_dir / f"photo-{photo_id}-{uuid.uuid4().hex}{extension}"
            try:
                last_transfer_update = 0.0

                def transfer_progress(
                    bytes_read: int,
                    *,
                    item_index: int = index,
                    item_path: str = relative_path,
                ) -> None:
                    nonlocal last_transfer_update
                    now = time.monotonic()
                    if now - last_transfer_update < 5:
                        return
                    last_transfer_update = now
                    self._set_progress(
                        job_id,
                        "analyzing",
                        item_index - 1,
                        total,
                        f"Reading {item_index:,}/{total:,}: {Path(item_path).name} "
                        f"({bytes_read / 1_048_576:.1f} MB)",
                    )

                file_hash, _bytes_read = source.copy_to_local_and_hash(
                    relative_path,
                    temporary_path,
                    max_bytes=max_bytes,
                    progress=transfer_progress,
                )
                result = self._analyzer.analyze(
                    temporary_path,
                    source_mtime,
                    thumbnail_path,
                    preview_path,
                    source_name=relative_path,
                )
                city, region, country = self._place_resolver.resolve(
                    result.latitude, result.longitude
                )
                with session_scope() as session:
                    photo = session.get(Photo, photo_id)
                    if not photo:
                        continue
                    photo.file_hash = file_hash
                    photo.perceptual_hash = result.perceptual_hash
                    photo.difference_hash = result.difference_hash
                    photo.visual_features = result.visual_features
                    photo.width = result.width
                    photo.height = result.height
                    photo.capture_at = result.capture_at
                    photo.capture_source = result.capture_source
                    photo.camera_make = result.camera_make
                    photo.camera_model = result.camera_model
                    photo.lens_model = result.lens_model
                    photo.iso = result.iso
                    photo.exposure_time = result.exposure_time
                    photo.aperture = result.aperture
                    photo.focal_length = result.focal_length
                    photo.latitude = result.latitude
                    photo.longitude = result.longitude
                    photo.altitude = result.altitude
                    photo.place_city = city
                    photo.place_region = region
                    photo.place_country = country
                    photo.sharpness_score = result.sharpness_score
                    photo.exposure_score = result.exposure_score
                    photo.contrast_score = result.contrast_score
                    photo.color_score = result.color_score
                    photo.resolution_score = result.resolution_score
                    photo.quality_score = result.quality_score
                    photo.face_count = len(result.faces)
                    photo.thumb_path = thumb_relative
                    photo.preview_path = preview_relative
                    photo.analysis_status = "done"
                    photo.analysis_version = self.settings.analysis_version
                    photo.analysis_attempts = 0
                    photo.analysis_error = None
                    photo.analyzed_at = utcnow()
                    for old_face in list(photo.faces):
                        session.delete(old_face)
                    for face in result.faces:
                        photo.faces.append(
                            Face(
                                x=face.x,
                                y=face.y,
                                width=face.width,
                                height=face.height,
                                confidence=face.confidence,
                                embedding=face.embedding,
                            )
                        )
                    self._set_job_in_session(
                        session,
                        job_id,
                        "analyzing",
                        index,
                        total,
                        f"Analyzed {index:,}/{total:,}: {Path(relative_path).name}",
                    )
            except (ImageTooLargeError, Image.DecompressionBombError, SourceSizeLimitError) as exc:
                logger.warning("Image exceeds a configured safety limit %s: %s", relative_path, exc)
                self._record_analysis_error(photo_id, str(exc), status="too_large")
            except UnidentifiedImageError as exc:
                logger.warning("Unsupported/corrupt image %s: %s", relative_path, exc)
                self._record_analysis_error(photo_id, str(exc), status="unsupported")
            except Exception as exc:
                logger.exception("Analysis failed for %s", relative_path)
                self._record_analysis_error(photo_id, str(exc), status="error")
            finally:
                temporary_path.unlink(missing_ok=True)

            if index % 10 == 0:
                self._set_progress(
                    job_id,
                    "analyzing",
                    index,
                    total,
                    f"Analyzed {index:,}/{total:,} photos",
                )

    def _record_analysis_error(self, photo_id: int, error: str, status: str) -> None:
        with session_scope() as session:
            photo = session.get(Photo, photo_id)
            if not photo:
                return
            photo.analysis_status = status
            if status in {"unsupported", "too_large"}:
                photo.analysis_version = self.settings.analysis_version
            photo.analysis_error = error[:2000]

    def _build_collections(self, job_id: int) -> None:
        self._set_progress(job_id, "curating", 0, 0, "Preparing collection heuristics")
        with session_scope() as session:
            specs = generate_collection_specs(
                session,
                self.settings,
                progress=lambda message: self._set_job_in_session(
                    session, job_id, "curating", 0, 0, message
                ),
            )
            total = len(specs)
            persist_collection_specs(
                session,
                specs,
                self.settings,
                progress=lambda current, count, title: self._set_job_in_session(
                    session,
                    job_id,
                    "curating",
                    current,
                    count,
                    f"Saving collection: {title}",
                ),
            )
            self._set_state_in_session(session, "last_curation_at", datetime.now(UTC).isoformat())
        self._set_progress(
            job_id,
            "curating",
            total,
            total,
            f"Built {total:,} suggested collections",
        )

    def _check_cancelled(self, job_id: int) -> None:
        if self._stop_event.is_set():
            raise WorkerStopping("Application is stopping")
        with session_scope() as session:
            requested = session.scalar(select(Job.cancel_requested).where(Job.id == job_id))
        if requested:
            raise JobCancelled("Cancellation requested")

    def _set_progress(
        self, job_id: int, phase: str, current: int, total: int, message: str
    ) -> None:
        with session_scope() as session:
            self._set_job_in_session(session, job_id, phase, current, total, message)

    @staticmethod
    def _set_job_in_session(
        session, job_id: int, phase: str, current: int, total: int, message: str
    ) -> None:
        job = session.get(Job, job_id)
        if not job:
            return
        job.phase = phase
        job.progress_current = current
        job.progress_total = total
        job.message = message
        job.heartbeat_at = utcnow()

    @staticmethod
    def _set_state_in_session(session, key: str, value: str) -> None:
        state = session.get(AppState, key)
        if state is None:
            state = AppState(key=key, value=value)
            session.add(state)
        else:
            state.value = value

    def _requeue_job(self, job_id: int, message: str) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            job.status = "queued"
            # Keep the current phase so _execute_job can resume at the next
            # incomplete stage rather than restarting a full job from scan.
            job.message = message
            job.error = None
            job.started_at = None
            job.heartbeat_at = utcnow()

    def _finish_job(
        self, job_id: int, status: str, message: str, error: str | None = None
    ) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            job.status = status
            job.phase = status
            job.message = message
            job.error = error[:4000] if error else None
            job.finished_at = utcnow()
            job.heartbeat_at = utcnow()

    def _schedule_periodic_scan_if_due(self) -> None:
        if not self.settings.auto_start or self.settings.scan_interval_hours <= 0:
            return
        with session_scope() as session:
            active = session.scalar(
                select(func.count()).select_from(Job).where(Job.status.in_(("queued", "running")))
            )
            state = session.get(AppState, "last_scan_at")
            latest_attempt = session.scalar(
                select(func.max(Job.created_at)).where(Job.kind.in_(("full", "scan")))
            )
        if active:
            return
        due = True
        reference_times: list[datetime] = []
        if state:
            try:
                reference_times.append(datetime.fromisoformat(state.value))
            except ValueError:
                pass
        if latest_attempt is not None:
            reference_times.append(latest_attempt)
        # Use the latest success or attempt. Otherwise a currently unreachable
        # source with an older successful scan would trigger a tight failure loop.
        normalized = [
            value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            for value in reference_times
        ]
        if normalized:
            due = datetime.now(UTC) - max(normalized) >= timedelta(
                hours=self.settings.scan_interval_hours
            )
        if due:
            self.enqueue("full")
            time.sleep(1)

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image, ImageDraw
from sqlalchemy import func, select

from app.database import init_database, session_scope
from app.models import AppState, Collection, Job, Photo
from app.source import SourceEntry
from app.worker import WorkerService, WorkerStopping


def test_recovery_preserves_attempt_count_and_requeues_jobs(settings):
    init_database(settings.database_url)
    with session_scope() as session:
        session.add(Job(kind="analysis", status="running", phase="analysis"))
        session.add(
            Photo(
                relative_path="interrupted.jpg",
                file_name="interrupted.jpg",
                extension=".jpg",
                size_bytes=10,
                source_mtime=1.0,
                active=True,
                analysis_status="analyzing",
                analysis_attempts=2,
            )
        )

    stale_original = settings.temp_dir / "orphan.jpg"
    stale_derivative = settings.thumbs_dir / "aa" / "orphan.jpg.tmp"
    stale_original.write_bytes(b"partial")
    stale_derivative.parent.mkdir(parents=True, exist_ok=True)
    stale_derivative.write_bytes(b"partial")

    worker = WorkerService(settings)
    worker._recover_interrupted_work()
    worker._cleanup_stale_temp_files()
    assert not stale_original.exists()
    assert not stale_derivative.exists()

    with session_scope() as session:
        job = session.scalar(select(Job))
        photo = session.scalar(select(Photo))
        assert job is not None and job.status == "queued"
        assert photo is not None and photo.analysis_status == "error"
        assert photo.analysis_attempts == 2
        assert "interrupted" in (photo.analysis_error or "").lower()

    worker._stop_event.set()
    with pytest.raises(WorkerStopping):
        worker._check_cancelled(job.id)


def test_partial_scan_never_deactivates_unseen_photos(settings):
    init_database(settings.database_url)
    with session_scope() as session:
        session.add(
            Photo(
                relative_path="existing.jpg",
                file_name="existing.jpg",
                extension=".jpg",
                size_bytes=10,
                source_mtime=1,
                active=True,
            )
        )
        job = Job(kind="scan", status="running", phase="scanning")
        session.add(job)
        session.flush()
        job_id = job.id

    class ScanSource:
        def __init__(self, warnings: int):
            self.walk_warnings = warnings

        def walk(self):
            yield SourceEntry("new.jpg", "new.jpg", 20, 2)

    worker = WorkerService(settings)
    worker._scan(job_id, ScanSource(warnings=1))  # type: ignore[arg-type]
    with session_scope() as session:
        assert session.scalar(
            select(Photo.active).where(Photo.relative_path == "existing.jpg")
        ) is True

    worker._scan(job_id, ScanSource(warnings=0))  # type: ignore[arg-type]
    with session_scope() as session:
        assert session.scalar(
            select(Photo.active).where(Photo.relative_path == "existing.jpg")
        ) is False


def test_recent_failed_scan_throttles_automatic_retry(settings):
    init_database(settings.database_url)
    with session_scope() as session:
        session.add(
            AppState(
                key="last_scan_at",
                value=(datetime.now(UTC) - timedelta(days=3)).isoformat(),
            )
        )
        session.add(Job(kind="full", status="failed", phase="failed"))

    worker = WorkerService(settings)
    worker._schedule_periodic_scan_if_due()

    with session_scope() as session:
        queued = session.scalar(
            select(func.count()).select_from(Job).where(Job.status == "queued")
        )
        assert queued == 0


def test_full_job_is_not_suppressed_by_smaller_queued_job(settings):
    init_database(settings.database_url)
    worker = WorkerService(settings)

    smaller_id = worker.enqueue("collections")
    full_id = worker.enqueue("full")

    assert full_id != smaller_id
    assert worker.enqueue("full") == full_id


def test_full_job_scans_analyzes_and_curates_local_library(settings):
    start = datetime(2023, 7, 14, 10, 0)
    for index in range(6):
        image = Image.new("RGB", (640, 420), (50 + index * 20, 100, 150))
        ImageDraw.Draw(image).rectangle((80 + index * 3, 70, 500, 350), outline="white", width=8)
        exif = Image.Exif()
        exif[36867] = (start + timedelta(minutes=index * 5)).strftime("%Y:%m:%d %H:%M:%S")
        image.save(settings.photo_root / f"photo-{index}.jpg", exif=exif, quality=90)

    init_database(settings.database_url)
    worker = WorkerService(settings)
    job_id = worker.enqueue("full")
    assert worker._claim_next_job() == job_id
    worker._execute_job(job_id)

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.status == "completed"
        assert session.scalar(select(func.count()).select_from(Photo)) == 6
        assert (
            session.scalar(
                select(func.count()).select_from(Photo).where(Photo.analysis_status == "done")
            )
            == 6
        )
        assert session.scalar(select(func.count()).select_from(Collection)) >= 2
        assert all(photo.thumb_path for photo in session.scalars(select(Photo)))

    assert not list(settings.temp_dir.iterdir())

from __future__ import annotations

import hmac
import logging
import os
import re
import shutil
import time
from collections import deque
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session
from starlette.types import ASGIApp, Receive, Scope, Send

from app import __version__
from app.config import Settings
from app.database import database_is_healthy, session_factory
from app.models import Collection, CollectionPhoto, Face, Job, Photo
from app.source import build_source
from app.worker import WorkerService

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
ISSUE_STATES = ("error", "unsupported", "too_large", "pending", "analyzing")


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _age_seconds(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return max(0, int((datetime.now(UTC) - value.astimezone(UTC)).total_seconds()))


class DiagnosticSanitizer:
    """Remove deployment secrets and infrastructure identifiers from diagnostics."""

    def __init__(self, settings: Settings) -> None:
        replacements = {
            settings.mcp_bearer_token.get_secret_value(): "<mcp-token>",
            settings.smb_password.get_secret_value(): "<smb-password>",
            settings.database_url: "<database>",
            str(settings.data_dir): "<data-dir>",
            str(settings.photo_root): "<photo-root>",
            settings.smb_host: "<smb-host>",
            settings.smb_share: "<smb-share>",
            settings.smb_user: "<smb-user>",
            settings.smb_domain: "<smb-domain>",
            settings.smb_path: "<smb-path>",
        }
        self._replacements = sorted(
            ((value, replacement) for value, replacement in replacements.items() if len(value) >= 3),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    def clean(self, value: object | None, limit: int = 800) -> str | None:
        if value is None:
            return None
        result = str(value)
        for secret, replacement in self._replacements:
            result = result.replace(secret, replacement)
        result = re.sub(r"(?i)(password|token|secret)=([^\s,;]+)", r"\1=<redacted>", result)
        result = " ".join(result.split())
        if len(result) > limit:
            return f"{result[: limit - 1]}…"
        return result


class DiagnosticLogHandler(logging.Handler):
    """Bounded, sanitized warning/error ring for remote troubleshooting."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(level=logging.WARNING)
        self._entries: deque[dict[str, object]] = deque(maxlen=settings.mcp_log_entries)
        self._lock = Lock()
        self._sanitizer = DiagnosticSanitizer(settings)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry: dict[str, object] = {
                "at": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="seconds"),
                "level": record.levelname,
                "logger": record.name,
                "message": self._sanitizer.clean(record.getMessage(), limit=1000) or "",
            }
            with self._lock:
                self._entries.append(entry)
        except Exception:
            self.handleError(record)

    def snapshot(self, level: str, limit: int) -> list[dict[str, object]]:
        minimum = logging._nameToLevel[level.upper()]  # noqa: SLF001
        with self._lock:
            entries = list(self._entries)
        return [entry for entry in reversed(entries) if logging._nameToLevel[str(entry["level"])] >= minimum][
            :limit
        ]


class BearerTokenASGI:
    """Minimal optional bearer gate around only the MCP sub-application."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            authorization = dict(scope.get("headers", [])).get(b"authorization", b"")
            if not hmac.compare_digest(authorization, self._expected):
                body = b'{"error":"MCP bearer token required"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


@contextmanager
def _read_session() -> Any:
    session: Session = session_factory()()
    try:
        yield session
    finally:
        # Closing rolls back the read transaction without expiring already loaded
        # scalar attributes that bounded serializers use outside this context.
        session.close()


def _group_counts(session: Session, column: Any, *filters: Any) -> dict[str, int]:
    rows = session.execute(select(column, func.count()).where(*filters).group_by(column)).all()
    return {str(key): int(count) for key, count in rows}


def _job_data(job: Job, sanitizer: DiagnosticSanitizer) -> dict[str, object]:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "phase": job.phase,
        "progress": {"current": job.progress_current, "total": job.progress_total},
        "message": sanitizer.clean(job.message),
        "error": sanitizer.clean(job.error),
        "cancel_requested": job.cancel_requested,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "heartbeat_age_seconds": _age_seconds(job.heartbeat_at),
    }


def _path_stats(root: Path, stop_after: int = 5000) -> dict[str, object]:
    file_count = 0
    entry_count = 0
    size = 0
    stack = [root]
    try:
        while stack and entry_count < stop_after:
            directory = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    entry_count += 1
                    if entry.is_file(follow_symlinks=False):
                        file_count += 1
                        try:
                            size += entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            pass
                    elif entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    if entry_count >= stop_after:
                        break
    except OSError:
        return {
            "files": file_count,
            "bytes": size,
            "entries_scanned": entry_count,
            "truncated": entry_count >= stop_after or bool(stack),
            "readable": False,
        }
    return {
        "files": file_count,
        "bytes": size,
        "entries_scanned": entry_count,
        "truncated": entry_count >= stop_after or bool(stack),
        "readable": True,
    }


def create_diagnostic_mcp(
    settings: Settings,
    worker: WorkerService,
    log_buffer: DiagnosticLogHandler,
) -> MCPServer[Any]:
    """Build the strictly read-only diagnostic MCP server."""

    sanitizer = DiagnosticSanitizer(settings)
    server: MCPServer[Any] = MCPServer(
        name="photo-curator-diagnostics",
        title="Photo Book Curator diagnostics",
        description="Bounded read-only operational diagnostics for a deployed photo curator.",
        instructions=(
            "Use overview first. All tools are read-only and return bounded summaries. "
            "No tool can access original image bytes, previews, credentials, arbitrary files, "
            "arbitrary SQL, or mutate jobs and review decisions. Request relative paths only when "
            "they are necessary to identify a failing library item."
        ),
        version=__version__,
        log_level="WARNING",
    )

    @server.tool(
        title="Deployment overview",
        annotations=READ_ONLY,
    )
    def get_overview() -> dict[str, Any]:
        """Return a compact first-pass health, workload, library and safe-config summary."""
        with _read_session() as session:
            photo_states = _group_counts(
                session,
                Photo.analysis_status,
                Photo.active.is_(True),
            )
            total_photos = int(session.scalar(select(func.count()).select_from(Photo)) or 0)
            active_photos = int(
                session.scalar(select(func.count()).select_from(Photo).where(Photo.active.is_(True)))
                or 0
            )
            located = int(
                session.scalar(
                    select(func.count()).select_from(Photo).where(
                        Photo.active.is_(True), Photo.latitude.is_not(None)
                    )
                )
                or 0
            )
            with_faces = int(
                session.scalar(
                    select(func.count()).select_from(Photo).where(
                        Photo.active.is_(True), Photo.face_count > 0
                    )
                )
                or 0
            )
            collection_kinds = _group_counts(session, Collection.kind)
            collection_statuses = _group_counts(session, Collection.status)
            decisions = _group_counts(session, CollectionPhoto.decision)
            active_job = session.scalar(
                select(Job).where(Job.status.in_(("queued", "running"))).order_by(Job.id)
            )

        disk: dict[str, int] | None = None
        try:
            usage = shutil.disk_usage(settings.data_dir)
            disk = {"total_bytes": usage.total, "free_bytes": usage.free}
        except OSError:
            pass

        warnings: list[str] = []
        issue_count = sum(photo_states.get(state, 0) for state in ISSUE_STATES)
        if issue_count:
            warnings.append(f"{issue_count} active photos are not in the done state")
        if active_job and (heartbeat_age := _age_seconds(active_job.heartbeat_at)) and heartbeat_age > 300:
            warnings.append(f"active job heartbeat is {heartbeat_age} seconds old")
        if not database_is_healthy():
            warnings.append("database health query failed")
        if not (worker._thread and worker._thread.is_alive()):  # noqa: SLF001
            warnings.append("background worker thread is not running")

        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "application": {
                "name": settings.app_name,
                "version": __version__,
                "database_healthy": database_is_healthy(),
                "worker_running": bool(worker._thread and worker._thread.is_alive()),  # noqa: SLF001
                "source_mode": settings.source_mode,
                "source_access": "read-only application interface",
            },
            "active_job": _job_data(active_job, sanitizer) if active_job else None,
            "library": {
                "photos_total": total_photos,
                "photos_active": active_photos,
                "photos_inactive": total_photos - active_photos,
                "analysis_states": photo_states,
                "with_faces": with_faces,
                "with_gps": located,
            },
            "collections": {
                "total": sum(collection_kinds.values()),
                "by_kind": collection_kinds,
                "by_status": collection_statuses,
                "decisions": decisions,
            },
            "safe_config": {
                "auto_start": settings.auto_start,
                "scan_interval_hours": settings.scan_interval_hours,
                "scan_max_files": settings.scan_max_files,
                "worker_batch_size": settings.worker_batch_size,
                "max_analysis_attempts": settings.max_analysis_attempts,
                "max_file_mb": settings.max_file_mb,
                "max_image_megapixels": settings.max_image_megapixels,
                "face_analysis": settings.face_analysis,
                "analysis_version": settings.analysis_version,
                "collection_version": settings.collection_version,
                "trip_home_radius_km": settings.trip_home_radius_km,
                "trip_min_distance_km": settings.trip_min_distance_km,
                "trip_max_days": settings.trip_max_days,
                "trip_context_days": settings.trip_context_days,
                "supported_extensions": sorted(settings.extension_set),
                "mcp_bearer_enabled": bool(settings.mcp_bearer_token.get_secret_value()),
                "mcp_host_validation_enabled": bool(settings.mcp_allowed_host_list),
            },
            "data_volume": disk,
            "warnings": warnings,
        }

    @server.tool(title="Source connectivity", annotations=READ_ONLY)
    def check_source_connection() -> dict[str, Any]:
        """Test configured read-only source authentication and listing without reading a photo."""
        started = time.monotonic()
        try:
            source = build_source(settings)
            message = source.test_connection()
            return {
                "ok": True,
                "source_mode": settings.source_mode,
                "message": sanitizer.clean(message),
                "duration_ms": round((time.monotonic() - started) * 1000),
                "source_write_api_available": False,
            }
        except Exception as exc:
            return {
                "ok": False,
                "source_mode": settings.source_mode,
                "error": sanitizer.clean(exc),
                "duration_ms": round((time.monotonic() - started) * 1000),
                "source_write_api_available": False,
            }

    @server.tool(title="Recent jobs", annotations=READ_ONLY)
    def list_recent_jobs(
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        status: Literal["any", "queued", "running", "completed", "failed", "cancelled"] = "any",
        before_id: Annotated[int | None, Field(ge=1)] = None,
    ) -> dict[str, Any]:
        """List a bounded newest-first page of durable jobs; use before_id as the next-page cursor."""
        with _read_session() as session:
            query = select(Job)
            if status != "any":
                query = query.where(Job.status == status)
            if before_id is not None:
                query = query.where(Job.id < before_id)
            jobs = list(session.scalars(query.order_by(desc(Job.id)).limit(limit)))
        return {
            "items": [_job_data(job, sanitizer) for job in jobs],
            "next_before_id": jobs[-1].id if len(jobs) == limit else None,
        }

    @server.tool(title="Photo issue queue", annotations=READ_ONLY)
    def list_photo_issues(
        status: Literal["all", "error", "unsupported", "too_large", "pending", "analyzing"] = "all",
        limit: Annotated[int, Field(ge=1, le=50)] = 20,
        before_id: Annotated[int | None, Field(ge=1)] = None,
        include_paths: bool = False,
    ) -> dict[str, Any]:
        """Inspect a bounded page of non-done photos. Paths are omitted unless explicitly requested."""
        with _read_session() as session:
            counts = _group_counts(
                session,
                Photo.analysis_status,
                Photo.active.is_(True),
                Photo.analysis_status.in_(ISSUE_STATES),
            )
            query = select(Photo).where(Photo.active.is_(True))
            if status == "all":
                query = query.where(Photo.analysis_status.in_(ISSUE_STATES))
            else:
                query = query.where(Photo.analysis_status == status)
            if before_id is not None:
                query = query.where(Photo.id < before_id)
            photos = list(session.scalars(query.order_by(desc(Photo.id)).limit(limit)))
            items = []
            for photo in photos:
                item: dict[str, object] = {
                    "id": photo.id,
                    "status": photo.analysis_status,
                    "extension": photo.extension,
                    "size_bytes": photo.size_bytes,
                    "attempts": photo.analysis_attempts,
                    "analysis_version": photo.analysis_version,
                    "error": sanitizer.clean(photo.analysis_error),
                    "updated_at": _iso(photo.updated_at),
                }
                if include_paths:
                    item["relative_path"] = sanitizer.clean(photo.relative_path, limit=500)
                items.append(item)
        return {
            "counts": counts,
            "items": items,
            "next_before_id": photos[-1].id if len(photos) == limit else None,
        }

    @server.tool(title="Photo diagnostic", annotations=READ_ONLY)
    def get_photo_diagnostic(photo_id: Annotated[int, Field(ge=1)], include_path: bool = False) -> dict[str, Any]:
        """Return stored analysis metadata for one photo ID without reading the source or image derivatives."""
        with _read_session() as session:
            photo = session.get(Photo, photo_id)
            if photo is None:
                return {"found": False, "photo_id": photo_id}
            duplicate_count = 0
            if photo.file_hash:
                duplicate_count = int(
                    session.scalar(
                        select(func.count()).select_from(Photo).where(
                            Photo.active.is_(True),
                            Photo.file_hash == photo.file_hash,
                            Photo.id != photo.id,
                        )
                    )
                    or 0
                )
            collection_count = int(
                session.scalar(
                    select(func.count()).select_from(CollectionPhoto).where(
                        CollectionPhoto.photo_id == photo.id
                    )
                )
                or 0
            )
            face_clusters = list(
                session.scalars(
                    select(Face.cluster_id)
                    .where(Face.photo_id == photo.id, Face.cluster_id.is_not(None))
                    .distinct()
                )
            )
            result: dict[str, Any] = {
                "found": True,
                "id": photo.id,
                "active": photo.active,
                "extension": photo.extension,
                "size_bytes": photo.size_bytes,
                "analysis": {
                    "status": photo.analysis_status,
                    "version": photo.analysis_version,
                    "attempts": photo.analysis_attempts,
                    "error": sanitizer.clean(photo.analysis_error),
                    "analyzed_at": _iso(photo.analyzed_at),
                },
                "image": {"width": photo.width, "height": photo.height},
                "capture": {
                    "at": _iso(photo.capture_at),
                    "source": photo.capture_source,
                    "camera_make": photo.camera_make,
                    "camera_model": photo.camera_model,
                },
                "location": {
                    "has_gps": photo.latitude is not None,
                    "city": photo.place_city,
                    "region": photo.place_region,
                    "country": photo.place_country,
                },
                "quality": {
                    "overall": photo.quality_score,
                    "sharpness": photo.sharpness_score,
                    "exposure": photo.exposure_score,
                    "contrast": photo.contrast_score,
                    "color": photo.color_score,
                    "resolution": photo.resolution_score,
                    "manual_rating": photo.manual_rating,
                },
                "relationships": {
                    "faces": photo.face_count,
                    "face_cluster_ids": face_clusters,
                    "exact_duplicates": duplicate_count,
                    "collections": collection_count,
                },
                "derivatives_recorded": {
                    "thumbnail": bool(photo.thumb_path),
                    "preview": bool(photo.preview_path),
                },
            }
            if include_path:
                result["relative_path"] = sanitizer.clean(photo.relative_path, limit=500)
            return result

    @server.tool(title="Collection summary", annotations=READ_ONLY)
    def list_collections(
        limit: Annotated[int, Field(ge=1, le=50)] = 20,
        before_id: Annotated[int | None, Field(ge=1)] = None,
        kind: str = "",
        status: str = "",
    ) -> dict[str, Any]:
        """List bounded collection summaries and review decisions without returning photo records."""
        with _read_session() as session:
            query = select(Collection)
            if before_id is not None:
                query = query.where(Collection.id < before_id)
            if kind:
                query = query.where(Collection.kind == kind)
            if status:
                query = query.where(Collection.status == status)
            collections = list(session.scalars(query.order_by(desc(Collection.id)).limit(limit)))
            collection_ids = [collection.id for collection in collections]
            decision_rows = (
                session.execute(
                    select(
                        CollectionPhoto.collection_id,
                        CollectionPhoto.decision,
                        func.count(),
                    )
                    .where(CollectionPhoto.collection_id.in_(collection_ids))
                    .group_by(CollectionPhoto.collection_id, CollectionPhoto.decision)
                ).all()
                if collection_ids
                else []
            )
            decisions_by_collection: dict[int, dict[str, int]] = {}
            for collection_id, decision, count in decision_rows:
                decisions_by_collection.setdefault(collection_id, {})[decision] = int(count)

            items: list[dict[str, object]] = []
            for collection in collections:
                decisions = decisions_by_collection.get(collection.id, {})
                items.append(
                    {
                        "id": collection.id,
                        "kind": collection.kind,
                        "title": collection.title,
                        "status": collection.status,
                        "automatic": collection.automatic,
                        "algorithm_version": collection.algorithm_version,
                        "score": collection.score,
                        "starts_at": _iso(collection.starts_at),
                        "ends_at": _iso(collection.ends_at),
                        "photo_count": sum(decisions.values()),
                        "decisions": decisions,
                    }
                )
        return {
            "items": items,
            "next_before_id": collections[-1].id if len(collections) == limit else None,
        }

    @server.tool(title="Read-only deployment audit", annotations=READ_ONLY)
    def run_read_only_audit(
        derivative_sample: Annotated[int, Field(ge=0, le=5000)] = 500,
    ) -> dict[str, Any]:
        """Run bounded SQLite, worker, derivative, model and local-storage consistency checks."""
        findings: list[dict[str, str]] = []
        with _read_session() as session:
            quick_check = [str(row[0]) for row in session.execute(text("PRAGMA quick_check(1)")).fetchmany(5)]
            foreign_keys = [list(row) for row in session.execute(text("PRAGMA foreign_key_check")).fetchmany(20)]
            journal_mode = str(session.execute(text("PRAGMA journal_mode")).scalar() or "unknown")
            error_counts = _group_counts(
                session,
                Photo.analysis_status,
                Photo.active.is_(True),
                Photo.analysis_status.in_(("error", "unsupported", "too_large")),
            )
            active_jobs = list(
                session.scalars(
                    select(Job).where(Job.status.in_(("queued", "running"))).order_by(Job.id)
                )
            )
            derivative_query = (
                select(Photo)
                .where(Photo.active.is_(True), Photo.analysis_status == "done")
                .order_by(Photo.id)
                .limit(derivative_sample)
            )
            derivative_photos = list(session.scalars(derivative_query)) if derivative_sample else []

        if quick_check != ["ok"]:
            findings.append({"severity": "critical", "code": "database_check", "summary": sanitizer.clean(quick_check) or "database quick check failed"})
        if foreign_keys:
            findings.append({"severity": "error", "code": "foreign_keys", "summary": f"at least {len(foreign_keys)} foreign-key violations"})
        if journal_mode.casefold() != "wal":
            findings.append({"severity": "warning", "code": "journal_mode", "summary": f"SQLite journal mode is {journal_mode}, expected wal"})
        if not (worker._thread and worker._thread.is_alive()):  # noqa: SLF001
            findings.append({"severity": "error", "code": "worker_stopped", "summary": "background worker thread is not running"})
        for job in active_jobs:
            age = _age_seconds(job.heartbeat_at or job.started_at or job.created_at)
            if age is not None and age > 300:
                findings.append({"severity": "warning", "code": "stale_job", "summary": f"job {job.id} has had no heartbeat for {age} seconds"})
        if error_counts:
            findings.append({"severity": "warning", "code": "photo_errors", "summary": f"active analysis issues: {error_counts}"})

        missing_derivatives: list[dict[str, object]] = []
        for photo in derivative_photos:
            missing: list[str] = []
            if not photo.thumb_path or not (settings.cache_dir / photo.thumb_path).is_file():
                missing.append("thumbnail")
            if not photo.preview_path or not (settings.cache_dir / photo.preview_path).is_file():
                missing.append("preview")
            if missing and len(missing_derivatives) < 20:
                missing_derivatives.append({"photo_id": photo.id, "missing": missing})
        if missing_derivatives:
            findings.append({"severity": "warning", "code": "missing_derivatives", "summary": f"{len(missing_derivatives)} sampled photos have missing derivatives"})

        temp_stats = _path_stats(settings.temp_dir, stop_after=1000)
        if temp_stats["files"]:
            findings.append({"severity": "warning", "code": "temporary_files", "summary": f"temporary directory contains {temp_stats['files']} files"})

        models = {
            "face_detector_present": settings.face_detector_model.is_file(),
            "face_recognizer_present": settings.face_recognizer_model.is_file(),
        }
        if settings.face_analysis and not all(models.values()):
            findings.append({"severity": "warning", "code": "face_models", "summary": "one or more configured face models are unavailable; fallback detection may be used"})
        if not settings.mcp_bearer_token.get_secret_value():
            findings.append({"severity": "info", "code": "mcp_auth", "summary": "MCP has no bearer token; keep the application on a trusted private network"})
        if not findings:
            findings.append({"severity": "ok", "code": "healthy", "summary": "no issues found by bounded checks"})

        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "read_only": True,
            "database": {
                "quick_check": quick_check,
                "foreign_key_violation_examples": foreign_keys,
                "journal_mode": journal_mode,
            },
            "worker": {
                "running": bool(worker._thread and worker._thread.is_alive()),  # noqa: SLF001
                "active_jobs": [_job_data(job, sanitizer) for job in active_jobs],
            },
            "analysis_issue_counts": error_counts,
            "derivative_sample": {
                "checked": len(derivative_photos),
                "missing_examples": missing_derivatives,
                "sample_limited": derivative_sample,
            },
            "temporary_storage": temp_stats,
            "models": models,
            "data_directory_writable": os.access(settings.data_dir, os.W_OK),
            "source_mutation_api_available": False,
            "findings": findings,
        }

    @server.tool(title="Recent diagnostic logs", annotations=READ_ONLY)
    def get_recent_diagnostic_logs(
        level: Literal["warning", "error", "critical"] = "warning",
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """Return recent sanitized warning/error messages held in a bounded in-memory ring."""
        return {
            "minimum_level": level.upper(),
            "items": log_buffer.snapshot(level, limit),
            "note": "The ring resets when the container restarts and excludes raw tracebacks.",
        }

    return server


def build_mcp_http_app(server: MCPServer[Any], settings: Settings) -> ASGIApp:
    allowed_hosts = settings.mcp_allowed_host_list
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=bool(allowed_hosts),
        allowed_hosts=allowed_hosts,
        allowed_origins=[],
    )
    app: ASGIApp = server.streamable_http_app(
        streamable_http_path="/",
        json_response=True,
        stateless_http=True,
        max_request_body_size=1_048_576,
        transport_security=security,
    )
    token = settings.mcp_bearer_token.get_secret_value()
    return BearerTokenASGI(app, token) if token else app

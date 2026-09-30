from __future__ import annotations

import csv
import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import case, desc, func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from app import __version__
from app.config import Settings, get_settings
from app.database import database_is_healthy, get_db, init_database
from app.models import Collection, CollectionPhoto, Job, Photo, photo_to_dict
from app.worker import WorkerService

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
DbSession = Annotated[Session, Depends(get_db)]


class DecisionBody(BaseModel):
    decision: str = Field(pattern="^(pending|keep|reject)$")
    note: str | None = Field(default=None, max_length=1000)


class RatingBody(BaseModel):
    rating: int | None = Field(default=None, ge=1, le=5)


class CollectionBody(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    status: str | None = Field(
        default=None, pattern="^(unreviewed|reviewing|shortlisted|complete|hidden)$"
    )


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _format_datetime(value: datetime | None) -> str:
    return value.strftime("%b %-d, %Y %H:%M") if value else "Unknown"


def _format_bytes(value: int | None) -> str:
    if not value:
        return "0 B"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _safe_csv_text(value: str) -> str:
    """Prevent spreadsheet formula execution from paths or user-entered notes."""
    return f"'{value}" if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def create_app(settings: Settings | None = None, start_worker: bool = True) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_directories()
    _configure_logging(settings)
    init_database(settings.database_url)

    worker = WorkerService(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.worker = worker

    templates = Jinja2Templates(directory=BASE_DIR / "templates")
    asset_version = max(
        int((BASE_DIR / "static" / name).stat().st_mtime)
        for name in ("app.css", "app.js")
    )
    templates.env.filters["datetime"] = _format_datetime
    templates.env.filters["filesize"] = _format_bytes
    application.mount(
        "/static", StaticFiles(directory=BASE_DIR / "static"), name="static"
    )
    application.mount(
        "/media", StaticFiles(directory=settings.cache_dir, check_dir=False), name="media"
    )

    def render(request: Request, name: str, **context):
        base_context = {
            "request": request,
            "app_name": settings.app_name,
            "version": __version__,
            "asset_version": asset_version,
            "source_summary": settings.source_summary(),
        }
        base_context.update(context)
        return templates.TemplateResponse(request=request, name=name, context=base_context)

    @application.get("/healthz")
    def health() -> JSONResponse:
        healthy = database_is_healthy()
        return JSONResponse(
            {
                "status": "ok" if healthy else "error",
                "version": __version__,
                "database": healthy,
                "worker": bool(worker._thread and worker._thread.is_alive())
                if start_worker
                else "disabled",
            },
            status_code=200 if healthy else 503,
        )

    @application.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, db: DbSession):
        photo_count = db.scalar(
            select(func.count()).select_from(Photo).where(Photo.active.is_(True))
        ) or 0
        analyzed_count = db.scalar(
            select(func.count())
            .select_from(Photo)
            .where(Photo.active.is_(True), Photo.analysis_status == "done")
        ) or 0
        face_count = db.scalar(
            select(func.count())
            .select_from(Photo)
            .where(Photo.active.is_(True), Photo.face_count > 0)
        ) or 0
        located_count = db.scalar(
            select(func.count())
            .select_from(Photo)
            .where(Photo.active.is_(True), Photo.latitude.is_not(None))
        ) or 0
        collection_count = db.scalar(select(func.count()).select_from(Collection)) or 0
        keep_count = db.scalar(
            select(func.count())
            .select_from(CollectionPhoto)
            .where(CollectionPhoto.decision == "keep")
        ) or 0
        collections = list(
            db.scalars(
                select(Collection)
                .options(selectinload(Collection.cover_photo))
                .where(Collection.status != "hidden")
                .order_by(desc(Collection.score), desc(Collection.updated_at))
                .limit(12)
            )
        )
        jobs = list(db.scalars(select(Job).order_by(desc(Job.id)).limit(8)))
        active_job = db.scalar(
            select(Job)
            .where(Job.status.in_(("queued", "running")))
            .order_by(Job.id)
        )
        return render(
            request,
            "dashboard.html",
            photo_count=photo_count,
            analyzed_count=analyzed_count,
            face_count=face_count,
            located_count=located_count,
            collection_count=collection_count,
            keep_count=keep_count,
            collections=collections,
            jobs=jobs,
            active_job=active_job,
        )

    @application.get("/collections", response_class=HTMLResponse)
    def collections_page(
        request: Request,
        db: DbSession,
        kind: str | None = None,
        status: str | None = None,
        page: Annotated[int, Query(ge=1)] = 1,
        per_page: Annotated[int, Query(ge=20, le=120)] = 60,
    ):
        conditions = []
        if kind:
            conditions.append(Collection.kind == kind)
        if status:
            conditions.append(Collection.status == status)
        else:
            conditions.append(Collection.status != "hidden")
        total = db.scalar(
            select(func.count()).select_from(Collection).where(*conditions)
        ) or 0
        collections = list(
            db.scalars(
                select(Collection)
                .options(selectinload(Collection.cover_photo))
                .where(*conditions)
                .order_by(desc(Collection.score), desc(Collection.updated_at))
                .offset((page - 1) * per_page)
                .limit(per_page)
            )
        )
        collection_ids = [collection.id for collection in collections]
        counts = {
            row.collection_id: (row.total, row.kept, row.rejected)
            for row in db.execute(
                select(
                    CollectionPhoto.collection_id,
                    func.count(CollectionPhoto.id).label("total"),
                    func.sum(case((CollectionPhoto.decision == "keep", 1), else_=0)).label("kept"),
                    func.sum(case((CollectionPhoto.decision == "reject", 1), else_=0)).label("rejected"),
                )
                .where(CollectionPhoto.collection_id.in_(collection_ids))
                .group_by(CollectionPhoto.collection_id)
            )
        }
        kinds = list(db.scalars(select(Collection.kind).distinct().order_by(Collection.kind)))
        return render(
            request,
            "collections.html",
            collections=collections,
            counts=counts,
            kinds=kinds,
            selected_kind=kind,
            selected_status=status,
            page=page,
            pages=max(1, (total + per_page - 1) // per_page),
            total=total,
        )

    @application.get("/collections/{collection_id}", response_class=HTMLResponse)
    def collection_detail(
        collection_id: int,
        request: Request,
        db: DbSession,
        page: Annotated[int, Query(ge=1)] = 1,
        decision: Annotated[
            str | None, Query(pattern="^(pending|keep|reject)$")
        ] = None,
        per_page: Annotated[int, Query(ge=20, le=200)] = 80,
    ):
        collection = db.get(Collection, collection_id)
        if not collection:
            raise HTTPException(404, "Collection not found")
        link_query = (
            select(CollectionPhoto)
            .options(selectinload(CollectionPhoto.photo))
            .where(CollectionPhoto.collection_id == collection_id)
        )
        if decision:
            link_query = link_query.where(CollectionPhoto.decision == decision)
        total = db.scalar(
            select(func.count())
            .select_from(CollectionPhoto)
            .where(
                CollectionPhoto.collection_id == collection_id,
                *([CollectionPhoto.decision == decision] if decision else []),
            )
        ) or 0
        links = list(
            db.scalars(
                link_query.order_by(CollectionPhoto.rank)
                .offset((page - 1) * per_page)
                .limit(per_page)
            )
        )
        decisions = dict(
            db.execute(
                select(CollectionPhoto.decision, func.count(CollectionPhoto.id))
                .where(CollectionPhoto.collection_id == collection_id)
                .group_by(CollectionPhoto.decision)
            ).all()
        )
        return render(
            request,
            "collection_detail.html",
            collection=collection,
            links=links,
            page=page,
            pages=max(1, (total + per_page - 1) // per_page),
            total=total,
            decisions=decisions,
            decision_filter=decision,
        )

    @application.get("/photos", response_class=HTMLResponse)
    def photos_page(
        request: Request,
        db: DbSession,
        page: Annotated[int, Query(ge=1)] = 1,
        year: Annotated[int | None, Query(ge=1900, le=9998)] = None,
        faces: bool | None = None,
        located: bool | None = None,
        q: Annotated[str | None, Query(max_length=200)] = None,
        per_page: Annotated[int, Query(ge=20, le=200)] = 80,
    ):
        conditions = [Photo.active.is_(True)]
        if year:
            conditions.extend(
                (
                    Photo.capture_at >= datetime(year, 1, 1),
                    Photo.capture_at < datetime(year + 1, 1, 1),
                )
            )
        if faces is True:
            conditions.append(Photo.face_count > 0)
        if faces is False:
            conditions.append(Photo.face_count == 0)
        if located is True:
            conditions.append(Photo.latitude.is_not(None))
        if located is False:
            conditions.append(Photo.latitude.is_(None))
        if q:
            pattern = f"%{q}%"
            conditions.append(
                or_(
                    Photo.relative_path.ilike(pattern),
                    Photo.place_city.ilike(pattern),
                    Photo.place_country.ilike(pattern),
                    Photo.camera_model.ilike(pattern),
                )
            )
        total = db.scalar(select(func.count()).select_from(Photo).where(*conditions)) or 0
        photos = list(
            db.scalars(
                select(Photo)
                .where(*conditions)
                .order_by(Photo.capture_at.desc(), Photo.id.desc())
                .offset((page - 1) * per_page)
                .limit(per_page)
            )
        )
        years = [
            int(value)
            for value in db.scalars(
                select(func.strftime("%Y", Photo.capture_at))
                .where(Photo.active.is_(True), Photo.capture_at.is_not(None))
                .distinct()
                .order_by(desc(func.strftime("%Y", Photo.capture_at)))
            )
            if value
        ]
        return render(
            request,
            "photos.html",
            photos=photos,
            page=page,
            pages=max(1, (total + per_page - 1) // per_page),
            total=total,
            years=years,
            selected_year=year,
            faces=faces,
            located=located,
            q=q or "",
        )

    @application.get("/photos/{photo_id}", response_class=HTMLResponse)
    def photo_detail(photo_id: int, request: Request, db: DbSession):
        photo = db.scalar(
            select(Photo)
            .options(selectinload(Photo.faces), selectinload(Photo.collection_links).selectinload(CollectionPhoto.collection))
            .where(Photo.id == photo_id)
        )
        if not photo:
            raise HTTPException(404, "Photo not found")
        return render(request, "photo_detail.html", photo=photo)

    @application.get("/status", response_class=HTMLResponse)
    def status_page(request: Request, db: DbSession):
        jobs = list(db.scalars(select(Job).order_by(desc(Job.id)).limit(30)))
        status_counts = dict(
            db.execute(
                select(Photo.analysis_status, func.count(Photo.id))
                .where(Photo.active.is_(True))
                .group_by(Photo.analysis_status)
            ).all()
        )
        errors = list(
            db.scalars(
                select(Photo)
                .where(
                    Photo.active.is_(True),
                    Photo.analysis_status.in_(("error", "unsupported", "too_large")),
                )
                .order_by(desc(Photo.updated_at))
                .limit(30)
            )
        )
        return render(
            request,
            "status.html",
            jobs=jobs,
            status_counts=status_counts,
            errors=errors,
            settings=settings,
        )

    @application.get("/api/status")
    def api_status(db: DbSession):
        active_job = db.scalar(
            select(Job)
            .where(Job.status.in_(("queued", "running")))
            .order_by(Job.id)
        )
        counts = dict(
            db.execute(
                select(Photo.analysis_status, func.count(Photo.id))
                .where(Photo.active.is_(True))
                .group_by(Photo.analysis_status)
            ).all()
        )
        return {
            "job": _job_dict(active_job) if active_job else None,
            "photos": counts,
            "database": database_is_healthy(),
        }

    @application.post("/api/jobs/{kind}")
    def start_job(kind: str, request: Request):
        try:
            job_id = request.app.state.worker.enqueue(kind)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "job_id": job_id}

    @application.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int, request: Request):
        if not request.app.state.worker.request_cancel(job_id):
            raise HTTPException(409, "Job is not active")
        return {"ok": True}

    @application.post("/api/retry-errors")
    def retry_analysis_errors(request: Request, db: DbSession):
        result = db.execute(
            update(Photo)
            .where(Photo.active.is_(True), Photo.analysis_status == "error")
            .values(
                analysis_status="pending",
                analysis_attempts=0,
                analysis_error=None,
            )
        )
        # Make the reset visible before waking the worker's separate DB session.
        db.commit()
        job_id = request.app.state.worker.enqueue("analyze")
        return {"ok": True, "reset": result.rowcount or 0, "job_id": job_id}

    @application.patch("/api/collection-links/{link_id}")
    def update_decision(link_id: int, body: DecisionBody, db: DbSession):
        link = db.get(CollectionPhoto, link_id)
        if not link:
            raise HTTPException(404, "Collection photo not found")
        link.decision = body.decision
        if body.note is not None:
            link.note = body.note
        return {"ok": True, "decision": link.decision}

    @application.patch("/api/collections/{collection_id}")
    def update_collection(collection_id: int, body: CollectionBody, db: DbSession):
        collection = db.get(Collection, collection_id)
        if not collection:
            raise HTTPException(404, "Collection not found")
        if body.title is not None:
            title = body.title.strip()
            if not title:
                raise HTTPException(422, "Collection title cannot be blank")
            collection.title = title
            collection.automatic = False
        if body.status is not None:
            collection.status = body.status
        return {"ok": True, "title": collection.title, "status": collection.status}

    @application.post("/api/collections/{collection_id}/keep-top")
    def keep_top(
        collection_id: int,
        db: DbSession,
        count: Annotated[int, Form(ge=1, le=500)] = 30,
    ):
        links = list(
            db.scalars(
                select(CollectionPhoto)
                .where(CollectionPhoto.collection_id == collection_id)
                .order_by(CollectionPhoto.rank)
            )
        )
        if not links:
            raise HTTPException(404, "Collection not found or empty")
        for index, link in enumerate(links):
            if link.decision == "pending":
                link.decision = "keep" if index < count else "pending"
        return RedirectResponse(f"/collections/{collection_id}", status_code=303)

    @application.patch("/api/photos/{photo_id}/rating")
    def rate_photo(photo_id: int, body: RatingBody, db: DbSession):
        photo = db.get(Photo, photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")
        photo.manual_rating = body.rating
        return {"ok": True, "rating": photo.manual_rating}

    @application.get("/api/photos/{photo_id}")
    def api_photo(photo_id: int, db: DbSession):
        photo = db.get(Photo, photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")
        return photo_to_dict(photo)

    @application.get("/collections/{collection_id}/manifest.csv")
    def collection_manifest(collection_id: int, db: DbSession):
        collection = db.get(Collection, collection_id)
        if not collection:
            raise HTTPException(404, "Collection not found")
        links = list(
            db.scalars(
                select(CollectionPhoto)
                .options(selectinload(CollectionPhoto.photo))
                .where(
                    CollectionPhoto.collection_id == collection_id,
                    CollectionPhoto.decision != "reject",
                )
                .order_by(CollectionPhoto.rank)
            )
        )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "rank",
                "decision",
                "relative_path",
                "capture_time",
                "quality_score",
                "place",
                "reason",
                "note",
            ]
        )
        for link in links:
            photo = link.photo
            writer.writerow(
                [
                    link.rank,
                    link.decision,
                    _safe_csv_text(photo.relative_path),
                    photo.capture_at.isoformat() if photo.capture_at else "",
                    f"{photo.quality_score:.4f}" if photo.quality_score is not None else "",
                    _safe_csv_text(
                        ", ".join(
                            value for value in (photo.place_city, photo.place_country) if value
                        )
                    ),
                    _safe_csv_text(link.reason),
                    _safe_csv_text(link.note),
                ]
            )
        safe_name = "".join(character if character.isalnum() else "-" for character in collection.title).strip("-")[:80]
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_name or "collection"}.csv"'},
        )

    return application


def _job_dict(job: Job) -> dict[str, object]:
    total = job.progress_total
    percent = round(job.progress_current / total * 100, 1) if total else None
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "phase": job.phase,
        "current": job.progress_current,
        "total": total,
        "percent": percent,
        "message": job.message,
        "error": job.error,
    }


app = create_app()

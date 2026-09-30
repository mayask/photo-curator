from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(primary_key=True)
    relative_path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    extension: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_mtime: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    scan_token: Mapped[str | None] = mapped_column(String(36), index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    analysis_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )
    analysis_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    analysis_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    analysis_error: Mapped[str | None] = mapped_column(Text)
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    file_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    perceptual_hash: Mapped[str | None] = mapped_column(String(32), index=True)
    difference_hash: Mapped[str | None] = mapped_column(String(32))
    visual_features: Mapped[bytes | None] = mapped_column(LargeBinary)

    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    capture_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    capture_source: Mapped[str | None] = mapped_column(String(20))
    camera_make: Mapped[str | None] = mapped_column(String(100))
    camera_model: Mapped[str | None] = mapped_column(String(100))
    lens_model: Mapped[str | None] = mapped_column(String(150))
    iso: Mapped[int | None] = mapped_column(Integer)
    exposure_time: Mapped[str | None] = mapped_column(String(40))
    aperture: Mapped[float | None] = mapped_column(Float)
    focal_length: Mapped[float | None] = mapped_column(Float)

    latitude: Mapped[float | None] = mapped_column(Float, index=True)
    longitude: Mapped[float | None] = mapped_column(Float, index=True)
    altitude: Mapped[float | None] = mapped_column(Float)
    place_city: Mapped[str | None] = mapped_column(String(120), index=True)
    place_region: Mapped[str | None] = mapped_column(String(120))
    place_country: Mapped[str | None] = mapped_column(String(120), index=True)

    sharpness_score: Mapped[float | None] = mapped_column(Float)
    exposure_score: Mapped[float | None] = mapped_column(Float)
    contrast_score: Mapped[float | None] = mapped_column(Float)
    color_score: Mapped[float | None] = mapped_column(Float)
    resolution_score: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float | None] = mapped_column(Float, index=True)
    face_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    manual_rating: Mapped[int | None] = mapped_column(Integer)

    thumb_path: Mapped[str | None] = mapped_column(Text)
    preview_path: Mapped[str | None] = mapped_column(Text)

    faces: Mapped[list[Face]] = relationship(
        back_populates="photo", cascade="all, delete-orphan"
    )
    collection_links: Mapped[list[CollectionPhoto]] = relationship(
        back_populates="photo", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_photos_analysis_queue", "active", "analysis_status", "analysis_version"),
        Index("ix_photos_capture_quality", "capture_at", "quality_score"),
    )


class Face(Base):
    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(primary_key=True)
    photo_id: Mapped[int] = mapped_column(
        ForeignKey("photos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    cluster_id: Mapped[int | None] = mapped_column(Integer, index=True)

    photo: Mapped[Photo] = relationship(back_populates="faces")


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(180), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="unreviewed", index=True
    )
    automatic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    algorithm_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    cover_photo_id: Mapped[int | None] = mapped_column(
        ForeignKey("photos.id", ondelete="SET NULL")
    )
    starts_at: Mapped[datetime | None] = mapped_column(DateTime)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    photos: Mapped[list[CollectionPhoto]] = relationship(
        back_populates="collection",
        cascade="all, delete-orphan",
        order_by="CollectionPhoto.rank",
    )
    cover_photo: Mapped[Photo | None] = relationship(foreign_keys=[cover_photo_id])


class CollectionPhoto(Base):
    __tablename__ = "collection_photos"

    id: Mapped[int] = mapped_column(primary_key=True)
    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    photo_id: Mapped[int] = mapped_column(
        ForeignKey("photos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    decision: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True
    )
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")

    collection: Mapped[Collection] = relationship(back_populates="photos")
    photo: Mapped[Photo] = relationship(back_populates="collection_links")

    __table_args__ = (
        UniqueConstraint("collection_id", "photo_id", name="uq_collection_photo"),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="queued", index=True
    )
    phase: Mapped[str] = mapped_column(String(40), nullable=False, default="queued")
    progress_current: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    error: Mapped[str | None] = mapped_column(Text)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AppState(Base):
    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


def photo_to_dict(photo: Photo) -> dict[str, Any]:
    """Small serialization helper used by JSON API responses."""
    return {
        "id": photo.id,
        "path": photo.relative_path,
        "captured_at": photo.capture_at.isoformat() if photo.capture_at else None,
        "width": photo.width,
        "height": photo.height,
        "quality": photo.quality_score,
        "faces": photo.face_count,
        "place": ", ".join(
            item for item in (photo.place_city, photo.place_country) if item
        ),
        "status": photo.analysis_status,
    }

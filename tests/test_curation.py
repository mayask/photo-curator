from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from app.curation import (
    generate_collection_specs,
    persist_collection_specs,
    split_bursts,
    split_events,
)
from app.database import init_database, session_scope
from app.models import Collection, CollectionPhoto, Photo


def _photo(number: int, captured: datetime, quality: float = 0.7) -> Photo:
    return Photo(
        relative_path=f"trip/photo-{number:03}.jpg",
        file_name=f"photo-{number:03}.jpg",
        extension=".jpg",
        size_bytes=1000 + number,
        source_mtime=captured.timestamp(),
        active=True,
        analysis_status="done",
        analysis_version=1,
        capture_at=captured,
        quality_score=quality,
        sharpness_score=quality,
        exposure_score=0.8,
        contrast_score=0.7,
        color_score=0.6,
        resolution_score=0.9,
        perceptual_hash=f"{number:016x}",
        file_hash=f"hash-{number}",
        width=3000,
        height=2000,
    )


def test_time_event_and_burst_segmentation(settings):
    start = datetime(2024, 5, 3, 9, 0)
    photos = [_photo(index, start + timedelta(seconds=index * 10)) for index in range(5)]
    photos += [_photo(10, start + timedelta(hours=12))]

    events = split_events(photos, settings)
    bursts = split_bursts(photos, settings.burst_gap_seconds)

    assert [len(group) for group in events] == [5, 1]
    assert [len(group) for group in bursts] == [5]


def test_collection_generation_and_decisions_survive_rebuild(settings):
    init_database(settings.database_url)
    start = datetime(2024, 5, 3, 9, 0)
    with session_scope() as session:
        photos = [_photo(index, start + timedelta(seconds=index * 10), 0.5 + index / 30) for index in range(8)]
        photos += [
            _photo(20 + index, start + timedelta(days=2, minutes=index * 20), 0.65)
            for index in range(7)
        ]
        photos[1].file_hash = photos[0].file_hash
        session.add_all(photos)

    with session_scope() as session:
        specs = generate_collection_specs(session, settings)
        keys = {spec.key for spec in specs}
        assert "highlights:all" in keys
        assert "duplicates:exact" in keys
        assert any(key.startswith("event:") for key in keys)
        assert any(key.startswith("series:") for key in keys)
        persist_collection_specs(session, specs, settings)

    with session_scope() as session:
        collection = session.scalar(select(Collection).where(Collection.key == "highlights:all"))
        assert collection is not None
        first_link = session.scalar(
            select(CollectionPhoto)
            .where(CollectionPhoto.collection_id == collection.id)
            .order_by(CollectionPhoto.rank)
        )
        assert first_link is not None
        first_link.decision = "keep"
        kept_photo_id = first_link.photo_id

    with session_scope() as session:
        specs = generate_collection_specs(session, settings)
        persist_collection_specs(session, specs, settings)

    with session_scope() as session:
        collection = session.scalar(select(Collection).where(Collection.key == "highlights:all"))
        preserved = session.scalar(
            select(CollectionPhoto).where(
                CollectionPhoto.collection_id == collection.id,
                CollectionPhoto.photo_id == kept_photo_id,
            )
        )
        assert preserved is not None
        assert preserved.decision == "keep"

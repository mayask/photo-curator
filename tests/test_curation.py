from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from app.curation import (
    _duplicate_specs,
    _folder_specs,
    _together_specs,
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


def test_near_duplicates_are_found_across_distant_dates():
    first = _photo(1, datetime(2018, 1, 1), 0.6)
    second = _photo(2, datetime(2025, 6, 1), 0.8)
    first.id = 1
    second.id = 2
    first.perceptual_hash = "1234567890abcdef"
    second.perceptual_hash = "1234567890abcdee"
    first.file_hash = "different-a"
    second.file_hash = "different-b"

    specs = _duplicate_specs([first, second])
    near = next(spec for spec in specs if spec.key == "duplicates:near")

    assert {link.photo.relative_path for link in near.links} == {
        first.relative_path,
        second.relative_path,
    }
    assert "library-wide" in near.description


def test_source_folders_are_used_as_human_album_clues():
    photos = [_photo(index, datetime(2020, 6, index + 1)) for index in range(5)]
    for index, photo in enumerate(photos, 1):
        photo.id = index
        photo.relative_path = f"summer-trip/photo-{index}.jpg"

    specs = _folder_specs(photos)

    assert len(specs) == 1
    assert specs[0].title == "Summer Trip"
    assert specs[0].key.startswith("folder:")


def test_faces_seen_together_create_relationship_collection():
    photos = [_photo(index, datetime(2024, 2, index + 1), 0.7) for index in range(3)]
    for index, photo in enumerate(photos, 1):
        photo.id = index
    specs = _together_specs(
        {photo.id: photo for photo in photos},
        {11: [1, 2, 3], 22: [1, 2, 3]},
    )

    assert len(specs) == 1
    assert specs[0].key == "together:11:22"
    assert specs[0].kind == "visitor"
    assert len(specs[0].links) == 3


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
        collection.title = "Our hand-picked favorites"
        collection.automatic = False

    with session_scope() as session:
        specs = generate_collection_specs(session, settings)
        highlights = next(spec for spec in specs if spec.key == "highlights:all")
        highlights.links = [
            link for link in highlights.links if link.photo.id != kept_photo_id
        ]
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
        assert collection.title == "Our hand-picked favorites"

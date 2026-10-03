from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from app.curation import (
    _duplicate_specs,
    _folder_specs,
    _highlight_specs,
    _make_spec,
    _together_specs,
    _trip_specs,
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
        capture_source="exif",
        quality_score=quality,
        face_count=0,
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
        first_link.note = "Print this one"
        kept_photo_id = first_link.photo_id
        collection.status = "shortlisted"
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
        assert preserved.note == "Print this one"
        assert collection.status == "shortlisted"
        assert collection.title == "Our hand-picked favorites"


def _travel_photo(number: int, day: int, location: tuple[float, float]) -> Photo:
    photo = _photo(number, datetime(2024, 1, 1, 10) + timedelta(days=day, minutes=number % 100))
    photo.id = number
    photo.latitude, photo.longitude = location
    photo.place_city = "Approximate city"
    return photo


def test_trips_require_away_and_return_not_density_or_city_names(settings):
    home = (43.36, -8.41)
    away = (48.86, 2.35)
    photos = [_travel_photo(i + 1, i * 4, home) for i in range(7)]
    for day in range(27, 32):
        for _ in range(8):
            photos.append(_travel_photo(len(photos) + 1, day, away))
    photos.extend([_travel_photo(len(photos) + i + 1, 33 + i * 4, home) for i in range(7)])

    trips = _trip_specs(photos, settings)

    assert len(trips) == 1
    assert len(trips[0].links) == 40
    assert all(link.photo.latitude == away[0] for link in trips[0].links)
    assert "before departure" in trips[0].description
    assert "after return" in trips[0].description
    assert trips[0].starts_at.date() == datetime(2024, 1, 28).date()
    assert trips[0].ends_at.date() == datetime(2024, 2, 1).date()
    assert _trip_specs([_travel_photo(i + 1, i // 6, home) for i in range(300)], settings) == []


def test_trips_reject_moves_long_stays_and_unconfirmed_returns(settings):
    home = (43.36, -8.41)
    new_residence = (53.55, 10.0)
    before = [_travel_photo(i + 1, i * 4, home) for i in range(7)]
    move = [
        _travel_photo(100 + i, 27 + i // 6, new_residence)
        for i in range(240)
    ]
    returned = [_travel_photo(1000 + i, 70 + i * 4, home) for i in range(7)]
    assert _trip_specs(before + move + returned, settings) == []
    assert _trip_specs(before + move[:30], settings) == []
    # A single photo back at the old address is not enough to confirm a return
    # to routine life there (it could be a visit during a relocation).
    assert _trip_specs(before + move[:30] + returned[:1], settings) == []


def test_trip_base_is_time_local_after_changing_residences(settings):
    old_home = (43.36, -8.41)
    new_home = (53.55, 10.0)
    away = (41.9, 12.5)
    photos = [_travel_photo(i + 1, i * 4, old_home) for i in range(7)]
    photos += [_travel_photo(100 + i, 150 + i * 4, new_home) for i in range(8)]
    photos += [_travel_photo(200 + i, 184 + i // 6, away) for i in range(24)]
    photos += [_travel_photo(300 + i, 190 + i * 4, new_home) for i in range(7)]

    trips = _trip_specs(photos, settings)

    assert len(trips) == 1
    assert {link.photo.id for link in trips[0].links} == set(range(200, 224))


def test_trip_does_not_guess_from_missing_gps_or_file_modification_times(settings):
    photos = [_photo(i + 1, datetime(2024, 1, 1) + timedelta(days=i // 6)) for i in range(150)]
    for i, photo in enumerate(photos):
        photo.id = i + 1
    assert _trip_specs(photos, settings) == []
    for photo in photos:
        photo.capture_source = "mtime"
        photo.latitude, photo.longitude = (43.36, -8.41) if photo.id < 50 else (48.86, 2.35)
    assert _trip_specs(photos, settings) == []


def test_best_of_selection_keeps_quality_ranks_and_daily_diversity():
    photos = [_photo(i, datetime(2024, 6, 1) + timedelta(minutes=i), i / 100) for i in range(30)]
    photos += [_photo(100 + i, datetime(2024, 7, 1) + timedelta(minutes=i), 0.5) for i in range(8)]
    for photo in photos:
        photo.id = int(photo.file_name[6:9])
        photo.perceptual_hash = None
    best = next(spec for spec in _highlight_specs(photos) if spec.key == "highlights:year:2024")
    per_day = {}
    for link in best.links:
        day = link.photo.capture_at.date()
        per_day[day] = per_day.get(day, 0) + 1
    assert max(per_day.values()) <= 5
    assert [link.score for link in best.links] == sorted((link.score for link in best.links), reverse=True)
    assert "star ratings" in best.description
    assert "newest first" in best.description


def test_obsolete_trip_suggestions_keep_reviews_notes_and_renamed_titles(settings):
    init_database(settings.database_url)
    with session_scope() as session:
        photo = _photo(1, datetime(2024, 1, 1))
        session.add(photo)
        for key in ("untouched", "kept", "noted", "renamed"):
            collection = Collection(key=key, kind="trip", title=key, description="Old trip rule")
            session.add(CollectionPhoto(
                collection=collection, photo=photo,
                decision="keep" if key == "kept" else "pending",
                note="Compare this later" if key == "noted" else "",
            ))
            if key == "renamed":
                collection.automatic = False

    with session_scope() as session:
        persist_collection_specs(session, [], settings)

    with session_scope() as session:
        collections = {item.key: item for item in session.scalars(select(Collection))}
        assert set(collections) == {"kept", "noted", "renamed"}
        assert "Archived automatic suggestion" in collections["kept"].description
        assert "Archived automatic suggestion" in collections["noted"].description
        noted = session.scalar(select(CollectionPhoto).where(
            CollectionPhoto.collection_id == collections["noted"].id
        ))
        assert noted.note == "Compare this later"
        assert collections["renamed"].title == "renamed"


def test_retained_pending_note_keeps_photo_and_its_timeline_date(settings):
    init_database(settings.database_url)
    with session_scope() as session:
        old, new = _photo(1, datetime(2014, 1, 1)), _photo(2, datetime(2025, 1, 1))
        session.add_all((old, new))
        session.flush()
        persist_collection_specs(session, [_make_spec("favorites", "highlights", "Favorites", "", [old, new])], settings)
    with session_scope() as session:
        old = session.scalar(select(Photo).where(Photo.file_name == "photo-001.jpg"))
        new = session.scalar(select(Photo).where(Photo.file_name == "photo-002.jpg"))
        link = session.scalar(select(CollectionPhoto).where(CollectionPhoto.photo_id == old.id))
        link.note = "Still deciding"
        persist_collection_specs(session, [_make_spec("favorites", "highlights", "Favorites", "", [new])], settings)
    with session_scope() as session:
        collection = session.scalar(select(Collection))
        assert len(collection.photos) == 2
        assert collection.starts_at == datetime(2014, 1, 1)
        assert collection.ends_at == datetime(2025, 1, 1)
        assert any(link.note == "Still deciding" for link in collection.photos)

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Collection, CollectionPhoto, Face, Photo

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LinkCandidate:
    photo: Photo
    score: float
    reason: str


@dataclass(slots=True)
class CollectionSpec:
    key: str
    kind: str
    title: str
    description: str
    links: list[LinkCandidate]
    score: float = 0
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    cover_photo_id: int | None = None


class UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def haversine_km(
    latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
) -> float:
    earth_radius_km = 6371.0088
    lat_a, lon_a, lat_b, lon_b = map(
        math.radians, (latitude_a, longitude_a, latitude_b, longitude_b)
    )
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    )
    return earth_radius_km * 2 * math.asin(math.sqrt(value))


def hash_distance(left: str | None, right: str | None) -> int:
    if not left or not right:
        return 65
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 65


def photo_quality(photo: Photo) -> float:
    score = photo.quality_score if photo.quality_score is not None else 0.35
    if photo.manual_rating:
        score = score * 0.7 + (photo.manual_rating / 5) * 0.3
    return float(np.clip(score, 0, 1))


def photo_reason(photo: Photo, prefix: str = "") -> str:
    traits: list[str] = []
    if (photo.sharpness_score or 0) >= 0.72:
        traits.append("sharp")
    if (photo.exposure_score or 0) >= 0.78:
        traits.append("balanced light")
    if (photo.color_score or 0) >= 0.65:
        traits.append("strong color")
    if photo.face_count:
        traits.append(f"{photo.face_count} face{'s' if photo.face_count != 1 else ''}")
    if photo.latitude is not None:
        traits.append("has location")
    summary = ", ".join(traits[:3]) or "representative frame"
    return f"{prefix}: {summary}" if prefix else summary


def _time_label(start: datetime, end: datetime) -> str:
    if start.date() == end.date():
        return start.strftime("%b %-d, %Y")
    if start.year == end.year:
        return f"{start.strftime('%b %-d')} – {end.strftime('%b %-d, %Y')}"
    return f"{start.strftime('%b %-d, %Y')} – {end.strftime('%b %-d, %Y')}"


def _dominant_place(photos: Sequence[Photo]) -> str | None:
    places = [
        ", ".join(value for value in (photo.place_city, photo.place_country) if value)
        for photo in photos
    ]
    places = [place for place in places if place]
    return Counter(places).most_common(1)[0][0] if places else None


def _centroid(photos: Sequence[Photo]) -> tuple[float | None, float | None]:
    coordinates = [
        (photo.latitude, photo.longitude)
        for photo in photos
        if photo.latitude is not None and photo.longitude is not None
    ]
    if not coordinates:
        return None, None
    return (
        sum(item[0] for item in coordinates) / len(coordinates),
        sum(item[1] for item in coordinates) / len(coordinates),
    )


def _make_spec(
    key: str,
    kind: str,
    title: str,
    description: str,
    photos: Sequence[Photo],
    reason_prefix: str = "",
    chronological: bool = False,
) -> CollectionSpec:
    unique = {photo.id: photo for photo in photos}
    records = list(unique.values())
    if chronological:
        records.sort(key=lambda photo: (photo.capture_at or datetime.min, photo.id))
    else:
        records.sort(key=lambda photo: (photo_quality(photo), photo.id), reverse=True)
    links = [
        LinkCandidate(photo, photo_quality(photo), photo_reason(photo, reason_prefix))
        for photo in records
    ]
    start_values = [photo.capture_at for photo in records if photo.capture_at]
    latitude, longitude = _centroid(records)
    cover = max(records, key=photo_quality).id if records else None
    average_top = (
        float(np.mean(sorted((photo_quality(photo) for photo in records), reverse=True)[:10]))
        if records
        else 0
    )
    return CollectionSpec(
        key=key,
        kind=kind,
        title=title,
        description=description,
        links=links,
        score=min(1.0, average_top * 0.8 + min(len(records), 30) / 150),
        starts_at=min(start_values) if start_values else None,
        ends_at=max(start_values) if start_values else None,
        latitude=latitude,
        longitude=longitude,
        cover_photo_id=cover,
    )


def split_events(photos: Sequence[Photo], settings: Settings) -> list[list[Photo]]:
    ordered = sorted(
        (photo for photo in photos if photo.capture_at),
        key=lambda photo: (photo.capture_at, photo.id),
    )
    if not ordered:
        return []
    groups: list[list[Photo]] = [[ordered[0]]]
    for photo in ordered[1:]:
        previous = groups[-1][-1]
        assert photo.capture_at and previous.capture_at
        gap_hours = (photo.capture_at - previous.capture_at).total_seconds() / 3600
        should_split = gap_hours > settings.event_gap_hours
        if photo.capture_at.date() != previous.capture_at.date() and gap_hours > 3.5:
            should_split = True
        if (
            not should_split
            and gap_hours > 1
            and photo.latitude is not None
            and photo.longitude is not None
            and previous.latitude is not None
            and previous.longitude is not None
            and haversine_km(
                photo.latitude,
                photo.longitude,
                previous.latitude,
                previous.longitude,
            )
            > 180
        ):
            should_split = True
        if should_split:
            groups.append([photo])
        else:
            groups[-1].append(photo)
    return groups


def split_bursts(photos: Sequence[Photo], gap_seconds: int) -> list[list[Photo]]:
    ordered = sorted(
        (photo for photo in photos if photo.capture_at),
        key=lambda photo: (photo.capture_at, photo.id),
    )
    groups: list[list[Photo]] = []
    current: list[Photo] = []
    for photo in ordered:
        if not current:
            current = [photo]
            continue
        assert photo.capture_at and current[-1].capture_at
        gap = (photo.capture_at - current[-1].capture_at).total_seconds()
        if 0 <= gap <= gap_seconds:
            current.append(photo)
        else:
            if len(current) >= 3:
                groups.append(current)
            current = [photo]
    if len(current) >= 3:
        groups.append(current)
    return groups


def _diverse_highlights(photos: Sequence[Photo], limit: int) -> list[Photo]:
    candidates = sorted(photos, key=lambda photo: (photo_quality(photo), photo.id), reverse=True)
    selected: list[Photo] = []
    seen_hashes: set[str] = set()
    per_day: Counter[date] = Counter()
    for candidate in candidates:
        if candidate.file_hash and candidate.file_hash in seen_hashes:
            continue
        day = candidate.capture_at.date() if candidate.capture_at else date.min
        if per_day[day] >= 5:
            continue
        too_similar = False
        for existing in selected[-40:]:
            if hash_distance(candidate.perceptual_hash, existing.perceptual_hash) <= 5:
                if candidate.capture_at and existing.capture_at:
                    gap = abs((candidate.capture_at - existing.capture_at).total_seconds())
                    if gap <= 600:
                        too_similar = True
                        break
        if too_similar:
            continue
        selected.append(candidate)
        if candidate.file_hash:
            seen_hashes.add(candidate.file_hash)
        per_day[day] += 1
        if len(selected) >= limit:
            break
    return selected


def _cluster_faces(session: Session) -> dict[int, list[int]]:
    faces = list(
        session.scalars(
            select(Face).where(Face.embedding.is_not(None)).order_by(Face.id)
        )
    )
    session.execute(delete(Face).where(False))  # force a predictable autoflush boundary
    if len(faces) < 3:
        for face in faces:
            face.cluster_id = None
        return {}

    valid_faces: list[Face] = []
    vectors: list[np.ndarray] = []
    expected_size: int | None = None
    for face in faces:
        vector = np.frombuffer(face.embedding or b"", dtype=np.float32)
        if not len(vector):
            continue
        if expected_size is None:
            expected_size = len(vector)
        if len(vector) != expected_size:
            continue
        norm = np.linalg.norm(vector)
        if not norm:
            continue
        valid_faces.append(face)
        vectors.append(vector / norm)

    if len(vectors) < 3:
        return {}
    labels = DBSCAN(eps=0.38, min_samples=3, metric="cosine", n_jobs=-1).fit_predict(
        np.vstack(vectors)
    )
    members: dict[int, list[Face]] = defaultdict(list)
    for face, label in zip(valid_faces, labels, strict=True):
        if label < 0:
            face.cluster_id = None
        else:
            members[int(label)].append(face)

    result: dict[int, list[int]] = {}
    for cluster_faces in members.values():
        stable_id = min(face.id for face in cluster_faces)
        for face in cluster_faces:
            face.cluster_id = stable_id
        result[stable_id] = [face.photo_id for face in cluster_faces]
    return result


def _event_specs(photos: Sequence[Photo], settings: Settings) -> list[CollectionSpec]:
    specs: list[CollectionSpec] = []
    events = split_events(photos, settings)
    qualifying = [event for event in events if len(event) >= 4]
    for event in qualifying:
        start = event[0].capture_at
        end = event[-1].capture_at
        if not start or not end:
            continue
        place = _dominant_place(event)
        date_label = _time_label(start, end)
        title = f"{place} — {date_label}" if place else f"Notable day — {date_label}"
        specs.append(
            _make_spec(
                f"event:{event[0].id}",
                "event",
                title,
                f"{len(event)} photos grouped by capture time"
                + (" and location." if any(p.latitude is not None for p in event) else "."),
                event,
                "event pick",
            )
        )

    # Multi-day, high-density runs become trip suggestions.
    dense_events = [event for event in qualifying if event[0].capture_at]
    current: list[Photo] = []
    trip_runs: list[list[Photo]] = []
    previous_day: date | None = None
    for event in dense_events:
        event_day = event[0].capture_at.date()  # type: ignore[union-attr]
        if previous_day is None or (event_day - previous_day).days <= 2:
            current.extend(event)
        else:
            if current:
                trip_runs.append(current)
            current = list(event)
        previous_day = event[-1].capture_at.date()  # type: ignore[union-attr]
        if current and (current[-1].capture_at.date() - current[0].capture_at.date()).days >= 14:  # type: ignore[union-attr]
            trip_runs.append(current)
            current = []
            previous_day = None
    if current:
        trip_runs.append(current)

    for run in trip_runs:
        start, end = run[0].capture_at, run[-1].capture_at
        if not start or not end:
            continue
        days = (end.date() - start.date()).days + 1
        if days < 2 or len(run) < max(12, days * 3):
            continue
        place = _dominant_place(run)
        title = f"Trip to {place}" if place else "Multi-day trip"
        title += f" — {_time_label(start, end)}"
        specs.append(
            _make_spec(
                f"trip:{run[0].id}",
                "trip",
                title,
                f"A dense {days}-day sequence with {len(run)} photos. Review the top-ranked frames for a photo-book chapter.",
                run,
                "trip highlight",
            )
        )
    return specs


def _duplicate_specs(photos: Sequence[Photo]) -> list[CollectionSpec]:
    specs: list[CollectionSpec] = []
    exact_groups: list[list[Photo]] = []
    hashes: dict[str, list[Photo]] = defaultdict(list)
    for photo in photos:
        if photo.file_hash:
            hashes[photo.file_hash].append(photo)
    exact_groups = [group for group in hashes.values() if len(group) > 1]
    if exact_groups:
        links: list[LinkCandidate] = []
        for number, group in enumerate(exact_groups, 1):
            for photo in sorted(group, key=lambda item: item.relative_path):
                links.append(
                    LinkCandidate(
                        photo,
                        photo_quality(photo),
                        f"byte-identical duplicate group {number}",
                    )
                )
        best = max((link.photo for link in links), key=photo_quality)
        specs.append(
            CollectionSpec(
                key="duplicates:exact",
                kind="duplicates",
                title="Exact duplicates",
                description=f"{len(exact_groups)} byte-identical groups. Nothing is deleted; use this as a review list.",
                links=links,
                score=1,
                cover_photo_id=best.id,
            )
        )

    candidates = sorted(
        (photo for photo in photos if photo.capture_at and photo.perceptual_hash),
        key=lambda photo: (photo.capture_at, photo.id),
    )
    union = UnionFind(photo.id for photo in candidates)
    left = 0
    exact_pairs = {photo.id: photo.file_hash for photo in candidates}
    for right, photo in enumerate(candidates):
        assert photo.capture_at
        while left < right:
            prior_time = candidates[left].capture_at
            assert prior_time
            if (photo.capture_at - prior_time).total_seconds() <= 600:
                break
            left += 1
        for prior in candidates[left:right]:
            if photo.file_hash and exact_pairs.get(prior.id) == photo.file_hash:
                continue
            if hash_distance(photo.perceptual_hash, prior.perceptual_hash) <= 7:
                union.union(photo.id, prior.id)
    near_groups: dict[int, list[Photo]] = defaultdict(list)
    for photo in candidates:
        near_groups[union.find(photo.id)].append(photo)
    groups = [group for group in near_groups.values() if len(group) > 1]
    if groups:
        links = []
        for number, group in enumerate(groups, 1):
            for photo in sorted(group, key=photo_quality, reverse=True):
                links.append(
                    LinkCandidate(
                        photo,
                        photo_quality(photo),
                        f"visually similar group {number}; best frame is ranked first",
                    )
                )
        specs.append(
            CollectionSpec(
                key="duplicates:near",
                kind="duplicates",
                title="Near duplicates",
                description=f"{len(groups)} groups detected with perceptual hashes within short time windows.",
                links=links,
                score=0.95,
                cover_photo_id=max((link.photo for link in links), key=photo_quality).id,
            )
        )
    return specs


def _burst_specs(photos: Sequence[Photo], settings: Settings) -> list[CollectionSpec]:
    groups = split_bursts(photos, settings.burst_gap_seconds)
    specs: list[CollectionSpec] = []
    for group in sorted(groups, key=len, reverse=True)[:100]:
        best = max(group, key=photo_quality)
        start = group[0].capture_at
        title = f"Series of {len(group)} — {start.strftime('%b %-d, %Y %H:%M') if start else 'unknown date'}"
        spec = _make_spec(
            f"series:{group[0].id}",
            "series",
            title,
            "Photos captured seconds apart. Highest-quality and most distinct frames are ranked first.",
            group,
            "series frame",
        )
        spec.cover_photo_id = best.id
        specs.append(spec)
    return specs


def _place_specs(photos: Sequence[Photo], settings: Settings) -> list[CollectionSpec]:
    located = [
        photo
        for photo in photos
        if photo.latitude is not None and photo.longitude is not None
    ]
    if len(located) < 5:
        return []
    coordinates = np.radians(
        np.array([(photo.latitude, photo.longitude) for photo in located], dtype=np.float64)
    )
    labels = DBSCAN(
        eps=settings.gps_cluster_km / 6371.0088,
        min_samples=5,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coordinates)
    groups: dict[int, list[Photo]] = defaultdict(list)
    for photo, label in zip(located, labels, strict=True):
        if label >= 0:
            groups[int(label)].append(photo)
    specs: list[CollectionSpec] = []
    for group in sorted(groups.values(), key=len, reverse=True)[:80]:
        place = _dominant_place(group) or "Mapped place"
        specs.append(
            _make_spec(
                f"place:{min(photo.id for photo in group)}",
                "place",
                place,
                f"{len(group)} photos within roughly {settings.gps_cluster_km:g} km, using embedded GPS metadata.",
                group,
                "place highlight",
            )
        )
    return specs


def _calendar_specs(photos: Sequence[Photo], settings: Settings) -> list[CollectionSpec]:
    specs: list[CollectionSpec] = []
    by_month_day: dict[tuple[int, int], list[Photo]] = defaultdict(list)
    for photo in photos:
        if photo.capture_at:
            by_month_day[(photo.capture_at.month, photo.capture_at.day)].append(photo)
    recurring = [
        group
        for group in by_month_day.values()
        if len(group) >= 8 and len({photo.capture_at.year for photo in group if photo.capture_at}) >= 2
    ]
    for group in sorted(recurring, key=len, reverse=True)[:40]:
        first = min(group, key=lambda photo: photo.capture_at or datetime.max)
        assert first.capture_at
        label = first.capture_at.strftime("%B %-d")
        years = sorted({photo.capture_at.year for photo in group if photo.capture_at})
        specs.append(
            _make_spec(
                f"recurring:{first.capture_at.strftime('%m-%d')}",
                "notable-date",
                f"Recurring date — {label}",
                f"Photos on this calendar date across {len(years)} years ({years[0]}–{years[-1]}). This may reveal an anniversary or tradition.",
                group,
                "recurring memory",
            )
        )

    if settings.holiday_country:
        try:
            import holidays

            years = sorted({photo.capture_at.year for photo in photos if photo.capture_at})
            calendar = holidays.country_holidays(
                settings.holiday_country,
                subdiv=settings.holiday_subdiv or None,
                years=years,
            )
            holiday_groups: dict[str, list[Photo]] = defaultdict(list)
            for photo in photos:
                if photo.capture_at and photo.capture_at.date() in calendar:
                    holiday_groups[str(calendar[photo.capture_at.date()])].append(photo)
            for name, group in holiday_groups.items():
                if len(group) < 3:
                    continue
                digest = hashlib.sha1(name.encode()).hexdigest()[:10]
                specs.append(
                    _make_spec(
                        f"holiday:{digest}",
                        "notable-date",
                        name,
                        f"Photos captured on {name}, grouped across the library.",
                        group,
                        "holiday memory",
                    )
                )
        except Exception as exc:
            logger.warning("Holiday calendar could not be loaded: %s", exc)
    return specs


def _face_specs(
    session: Session, photos_by_id: dict[int, Photo], cluster_photo_ids: dict[int, list[int]]
) -> list[CollectionSpec]:
    specs: list[CollectionSpec] = []
    for cluster_id, photo_ids in sorted(
        cluster_photo_ids.items(), key=lambda item: len(set(item[1])), reverse=True
    ):
        photos = [photos_by_id[photo_id] for photo_id in set(photo_ids) if photo_id in photos_by_id]
        if len(photos) < 3:
            continue
        ordered = sorted(
            (photo for photo in photos if photo.capture_at),
            key=lambda photo: photo.capture_at,
        )
        if not ordered:
            continue
        span_days = (ordered[-1].capture_at.date() - ordered[0].capture_at.date()).days  # type: ignore[union-attr]
        title = f"Person {cluster_id}"
        description = f"A locally clustered face appearing in {len(photos)} photos. Rename during review when person labels are added."
        kind = "people"
        if span_days <= 14 and len(photos) <= 40:
            title = f"Possible visitor — {ordered[0].capture_at.strftime('%b %Y')}"  # type: ignore[union-attr]
            description = "This face appears only in a short period, suggesting a visitor or someone newly met."
            kind = "visitor"
        specs.append(
            _make_spec(
                f"person:{cluster_id}",
                kind,
                title,
                description,
                photos,
                "face match",
            )
        )
    return specs[:100]


def _highlight_specs(photos: Sequence[Photo]) -> list[CollectionSpec]:
    specs: list[CollectionSpec] = []
    highlights = _diverse_highlights(photos, 150)
    if highlights:
        specs.append(
            _make_spec(
                "highlights:all",
                "highlights",
                "Print-worthy highlights",
                "A quality-ranked, de-duplicated selection with limits on near-identical frames and over-represented days.",
                highlights,
                "quality pick",
            )
        )

    by_year: dict[int, list[Photo]] = defaultdict(list)
    for photo in photos:
        if photo.capture_at:
            by_year[photo.capture_at.year].append(photo)
    for year, group in sorted(by_year.items(), reverse=True):
        if len(group) < 5:
            continue
        selected = _diverse_highlights(group, min(60, max(15, len(group) // 8)))
        specs.append(
            _make_spec(
                f"highlights:year:{year}",
                "highlights",
                f"Best of {year}",
                f"A diverse shortlist of {len(selected)} technically strong photos from {len(group)} analyzed in {year}.",
                selected,
                "year highlight",
            )
        )

    portraits = _diverse_highlights([photo for photo in photos if photo.face_count > 0], 100)
    if len(portraits) >= 5:
        specs.append(
            _make_spec(
                "highlights:portraits",
                "people",
                "Favorite people photos",
                "High-quality photos containing faces, with bursts and duplicates reduced.",
                portraits,
                "portrait pick",
            )
        )
    scenic = _diverse_highlights(
        [
            photo
            for photo in photos
            if photo.face_count == 0
            and photo.width is not None
            and photo.height is not None
            and photo.width > photo.height
        ],
        100,
    )
    if len(scenic) >= 5:
        specs.append(
            _make_spec(
                "highlights:scenic",
                "highlights",
                "Scenery and establishing shots",
                "Strong landscape-orientation photos without detected faces—useful for opening and transition pages.",
                scenic,
                "scenic pick",
            )
        )
    return specs


def _notable_day_specs(photos: Sequence[Photo]) -> list[CollectionSpec]:
    by_day: dict[date, list[Photo]] = defaultdict(list)
    for photo in photos:
        if photo.capture_at:
            by_day[photo.capture_at.date()].append(photo)
    by_year: dict[int, list[tuple[date, list[Photo]]]] = defaultdict(list)
    for day, group in by_day.items():
        by_year[day.year].append((day, group))
    specs: list[CollectionSpec] = []
    for year, groups in by_year.items():
        eligible = [item for item in groups if len(item[1]) >= 6]
        for day, group in sorted(eligible, key=lambda item: len(item[1]), reverse=True)[:8]:
            place = _dominant_place(group)
            title = f"Big day in {place}" if place else "A day worth remembering"
            title += f" — {day.strftime('%b %-d, %Y')}"
            specs.append(
                _make_spec(
                    f"notable-day:{day.isoformat()}",
                    "notable-date",
                    title,
                    f"One of {year}'s most photographed days, with {len(group)} images.",
                    group,
                    "day highlight",
                )
            )
    return specs


def generate_collection_specs(
    session: Session,
    settings: Settings,
    progress: Callable[[str], None] | None = None,
) -> list[CollectionSpec]:
    photos = list(
        session.scalars(
            select(Photo).where(
                Photo.active.is_(True),
                Photo.analysis_status == "done",
            )
        )
    )
    if not photos:
        return []
    photos_by_id = {photo.id: photo for photo in photos}
    if progress:
        progress("Clustering faces")
    face_clusters = _cluster_faces(session)
    builders: list[tuple[str, Callable[[], list[CollectionSpec]]]] = [
        ("Ranking highlights", lambda: _highlight_specs(photos)),
        ("Finding events and trips", lambda: _event_specs(photos, settings)),
        ("Finding notable days", lambda: _notable_day_specs(photos)),
        ("Finding places", lambda: _place_specs(photos, settings)),
        ("Finding duplicate candidates", lambda: _duplicate_specs(photos)),
        ("Finding bursts and series", lambda: _burst_specs(photos, settings)),
        ("Finding recurring and holiday dates", lambda: _calendar_specs(photos, settings)),
        ("Building people and visitor groups", lambda: _face_specs(session, photos_by_id, face_clusters)),
    ]
    specs: dict[str, CollectionSpec] = {}
    for message, builder in builders:
        if progress:
            progress(message)
        try:
            for spec in builder():
                if spec.links:
                    specs[spec.key] = spec
        except Exception:
            logger.exception("Collection heuristic failed: %s", message)
    return list(specs.values())


def persist_collection_specs(
    session: Session,
    specs: Sequence[CollectionSpec],
    settings: Settings,
    progress: Callable[[int, int, str], None] | None = None,
) -> int:
    existing = {
        collection.key: collection
        for collection in session.scalars(
            select(Collection).where(Collection.automatic.is_(True))
        )
    }
    desired_keys = {spec.key for spec in specs}
    total = len(specs)
    for index, spec in enumerate(specs, 1):
        collection = existing.get(spec.key)
        if collection is None:
            collection = Collection(key=spec.key, kind=spec.kind, title=spec.title)
            session.add(collection)
            session.flush()
        collection.kind = spec.kind
        collection.title = spec.title
        collection.description = spec.description
        collection.score = spec.score
        collection.algorithm_version = settings.collection_version
        collection.cover_photo_id = spec.cover_photo_id
        collection.starts_at = spec.starts_at
        collection.ends_at = spec.ends_at
        collection.latitude = spec.latitude
        collection.longitude = spec.longitude

        old_links = {link.photo_id: link for link in collection.photos}
        wanted_ids = {link.photo.id for link in spec.links}
        for photo_id, old_link in list(old_links.items()):
            if photo_id not in wanted_ids:
                session.delete(old_link)
        for rank, candidate in enumerate(spec.links, 1):
            link = old_links.get(candidate.photo.id)
            if link is None:
                link = CollectionPhoto(
                    collection=collection,
                    photo=candidate.photo,
                    decision="pending",
                )
                session.add(link)
            link.rank = rank
            link.score = candidate.score
            link.reason = candidate.reason
        if progress:
            progress(index, total, spec.title)
        if index % 20 == 0:
            session.flush()

    for key, collection in existing.items():
        if key not in desired_keys and collection.status == "unreviewed":
            session.delete(collection)
    session.flush()
    return total

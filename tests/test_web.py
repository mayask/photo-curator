from __future__ import annotations

from datetime import datetime, timedelta
from html import unescape

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database import init_database, session_scope
from app.main import create_app
from app.models import Collection, CollectionPhoto, Photo


def test_core_web_pages_and_health_render(settings):
    init_database(settings.database_url)
    application = create_app(settings, start_worker=False)
    with TestClient(application) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["database"] is True

        for path in ("/", "/collections", "/photos", "/status"):
            response = client.get(path)
            assert response.status_code == 200, response.text
            assert "Photo Book" in response.text

        # HTML select controls submit empty values for inactive filters.
        response = client.get("/photos?q=&year=&faces=true&located=")
        assert response.status_code == 200, response.text

        response = client.get("/api/status")
        assert response.status_code == 200
        assert response.json()["job"] is None


def test_review_mutations_and_manifest(settings):
    init_database(settings.database_url)
    with session_scope() as session:
        photo = Photo(
            relative_path="trip/favorite.jpg",
            file_name="favorite.jpg",
            extension=".jpg",
            size_bytes=123,
            source_mtime=1,
            analysis_status="done",
            quality_score=0.9,
        )
        collection = Collection(
            key="test:review",
            kind="highlights",
            title="Review me",
            description="Test collection",
        )
        link = CollectionPhoto(
            collection=collection,
            photo=photo,
            rank=1,
            score=0.9,
            reason="strong frame",
        )
        session.add_all((photo, collection, link))
        session.flush()
        photo_id, collection_id, link_id = photo.id, collection.id, link.id

    application = create_app(settings, start_worker=False)
    with TestClient(application) as client:
        response = client.patch(
            f"/api/collection-links/{link_id}",
            json={"decision": "keep", "note": "print large"},
        )
        assert response.status_code == 200
        assert response.json()["decision"] == "keep"

        response = client.patch(
            f"/api/collections/{collection_id}",
            json={"title": "Our favorites", "status": "shortlisted"},
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Our favorites"

        response = client.patch(
            f"/api/collections/{collection_id}", json={"title": "   "}
        )
        assert response.status_code == 422

        assert client.patch(
            f"/api/photos/{photo_id}/rating", json={"rating": 5}
        ).json()["rating"] == 5
        assert client.patch(
            f"/api/photos/{photo_id}/rating", json={"rating": None}
        ).json()["rating"] is None

        with session_scope() as session:
            failed_photo = session.get(Photo, photo_id)
            failed_photo.analysis_status = "error"
            failed_photo.analysis_attempts = 3
            failed_photo.analysis_error = "temporary source failure"
        retried = client.post("/api/retry-errors")
        assert retried.status_code == 200
        assert retried.json()["reset"] == 1

        manifest = client.get(f"/collections/{collection_id}/manifest.csv")
        assert manifest.status_code == 200
        assert "trip/favorite.jpg" in manifest.text
        assert "print large" in manifest.text

    with session_scope() as session:
        collection = session.get(Collection, collection_id)
        link = session.get(CollectionPhoto, link_id)
        photo = session.get(Photo, photo_id)
        assert collection is not None and collection.automatic is False
        assert link is not None and link.note == "print large"
        assert photo is not None and photo.analysis_status == "pending"
        assert photo.analysis_attempts == 0


def test_newest_first_views_group_by_month_without_changing_quality_rank(settings):
    application = create_app(settings, start_worker=False)
    with session_scope() as session:
        old_photo = Photo(
            relative_path="old.jpg", file_name="old.jpg", extension=".jpg",
            capture_at=datetime(2024, 1, 1), quality_score=0.95,
        )
        new_photo = Photo(
            relative_path="new.jpg", file_name="new.jpg", extension=".jpg",
            capture_at=datetime(2025, 2, 10), quality_score=0.5,
        )
        unknown_photo = Photo(
            relative_path="unknown.jpg", file_name="unknown.jpg", extension=".jpg",
            quality_score=0.9,
        )
        old_collection = Collection(
            key="old", kind="highlights", title="Old and highest score", score=1,
            starts_at=datetime(2023, 1, 1), ends_at=datetime(2023, 12, 31),
        )
        new_collection = Collection(
            key="new", kind="highlights", title="Newest collection", score=0.1,
            starts_at=datetime(2024, 1, 1), ends_at=datetime(2025, 2, 10),
        )
        undated_collection = Collection(key="undated", kind="event", title="Unknown date")
        session.add_all((old_photo, new_photo, unknown_photo, old_collection, new_collection, undated_collection))
        session.flush()
        old_id, new_id, unknown_id = old_photo.id, new_photo.id, unknown_photo.id
        old_collection_id, collection_id, undated_id = old_collection.id, new_collection.id, undated_collection.id
        session.add_all(
            CollectionPhoto(collection=new_collection, photo=photo, rank=rank, score=photo.quality_score)
            for photo, rank in ((old_photo, 1), (unknown_photo, 2), (new_photo, 3))
        )

    with TestClient(application) as client:
        for path in ("/", "/collections"):
            response = client.get(path)
            assert [item.id for item in response.context["collections"]] == [collection_id, old_collection_id, undated_id]
        collections = client.get("/collections")
        assert [section["key"] for section in collections.context["date_sections"]] == ["2025-02", "2023-12", "unknown"]
        assert "February 2025" in collections.text

        detail = client.get(f"/collections/{collection_id}")
        assert [link.photo_id for link in detail.context["links"]] == [new_id, old_id, unknown_id]
        assert [link.rank for link in detail.context["links"]] == [3, 1, 2]
        assert "Quality #3" in detail.text
        assert "January 2024" in detail.text
        assert "Unknown date" in detail.text

        library = client.get("/photos?group=year")
        assert [photo.id for photo in library.context["photos"]] == [new_id, old_id, unknown_id]
        assert [section["key"] for section in library.context["date_sections"]] == ["2025", "2024", "unknown"]
        for path in ("/collections", "/photos", f"/collections/{collection_id}"):
            flat = client.get(f"{path}?group=none")
            assert flat.status_code == 200
            assert 'class="date-divider"' not in flat.text
            assert client.get(f"{path}?group=decade").status_code == 422

        # "Keep best" remains quality-ranked, not first in the new date order.
        assert client.post(f"/api/collections/{collection_id}/keep-top", data={"count": 1}).status_code == 200
    with session_scope() as session:
        links = list(session.scalars(select(CollectionPhoto).order_by(CollectionPhoto.rank)))
        assert [(link.photo_id, link.decision) for link in links] == [
            (old_id, "keep"), (unknown_id, "pending"), (new_id, "pending")
        ]


def test_timeline_pagination_keeps_grouping_filters_and_page_size(settings):
    application = create_app(settings, start_worker=False)
    with session_scope() as session:
        collection = Collection(key="timeline", kind="highlights", title="Timeline")
        session.add(collection)
        for index in range(23):
            photo = Photo(
                relative_path=f"timeline/{index}.jpg", file_name=f"{index}.jpg", extension=".jpg",
                capture_at=datetime(2025, 1, 1) + timedelta(days=index),
                face_count=1, latitude=43,
            )
            session.add(CollectionPhoto(collection=collection, photo=photo, rank=index + 1))
            session.add(Collection(
                key=f"dated:{index}", kind="event", title=f"Day {index}",
                starts_at=photo.capture_at, ends_at=photo.capture_at,
            ))
        session.flush()
        collection_id = collection.id

    with TestClient(application) as client:
        for path in (
            "/collections?kind=event&group=year&per_page=20",
            "/photos?faces=true&located=true&year=2025&group=year&per_page=20",
            f"/collections/{collection_id}?decision=pending&group=year&per_page=20",
        ):
            first = client.get(path)
            second = client.get(path + "&page=2")
            assert first.status_code == second.status_code == 200
            assert first.context["grouping"] == second.context["grouping"] == "year"
            assert first.context["per_page"] == second.context["per_page"] == 20
            assert 'page=2&group=year&per_page=20' in unescape(first.text)
            field = "photos" if path.startswith("/photos?") else "collections" if path.startswith("/collections?") else "links"
            assert len(first.context[field]) == 20
            assert len(second.context[field]) == 3

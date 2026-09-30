from __future__ import annotations

from fastapi.testclient import TestClient

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

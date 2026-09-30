from __future__ import annotations

from fastapi.testclient import TestClient

from app.database import init_database
from app.main import create_app


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

        response = client.get("/api/status")
        assert response.status_code == 200
        assert response.json()["job"] is None

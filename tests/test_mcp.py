from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient
from mcp import Client
from pydantic import SecretStr

from app.database import session_scope
from app.main import create_app
from app.mcp_server import DiagnosticSanitizer
from app.models import Collection, CollectionPhoto, Job, Photo


def test_mcp_tools_are_read_only_and_bounded(settings):
    app = create_app(settings, start_worker=False)
    server = app.state.diagnostic_mcp
    with session_scope() as session:
        photo = Photo(
            relative_path="private/failed.jpg",
            file_name="failed.jpg",
            extension=".jpg",
            size_bytes=123,
            source_mtime=1.0,
            analysis_status="error",
            analysis_error="decoder failed",
        )
        collection = Collection(
            key="test",
            kind="highlights",
            title="Test highlights",
            description="Test",
        )
        analyzed = Photo(
            relative_path="private/analyzed.jpg",
            file_name="analyzed.jpg",
            extension=".jpg",
            size_bytes=456,
            source_mtime=2.0,
            analysis_status="done",
            thumb_path="thumbs/missing.jpg",
            preview_path="previews/missing.jpg",
        )
        session.add_all((photo, analyzed, collection, Job(kind="scan", status="completed")))
        session.flush()
        session.add(CollectionPhoto(collection_id=collection.id, photo_id=photo.id))
        photo_id = photo.id

    async def inspect_server():
        async with Client(server, raise_exceptions=True) as client:
            tools = await client.list_tools()
            overview = await client.call_tool("get_overview", {})
            source = await client.call_tool("check_source_connection", {})
            jobs = await client.call_tool("list_recent_jobs", {"limit": 2})
            issues = await client.call_tool("list_photo_issues", {"limit": 2})
            photo = await client.call_tool("get_photo_diagnostic", {"photo_id": photo_id})
            collections = await client.call_tool("list_collections", {"limit": 2})
            audit = await client.call_tool("run_read_only_audit", {"derivative_sample": 10})
            logs = await client.call_tool("get_recent_diagnostic_logs", {"limit": 2})
            return (
                tools.tools,
                overview.structured_content,
                source.structured_content,
                jobs.structured_content,
                issues.structured_content,
                photo.structured_content,
                collections.structured_content,
                audit.structured_content,
                logs.structured_content,
            )

    tools, overview, source, jobs, issues, photo, collections, audit, logs = asyncio.run(
        inspect_server()
    )

    assert {tool.name for tool in tools} == {
        "get_overview",
        "check_source_connection",
        "list_recent_jobs",
        "list_photo_issues",
        "get_photo_diagnostic",
        "list_collections",
        "run_read_only_audit",
        "get_recent_diagnostic_logs",
    }
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools)
    assert overview["application"]["database_healthy"] is True
    assert overview["library"]["photos_active"] == 2
    assert overview["library"]["analysis_states"] == {"done": 1, "error": 1}
    assert source["ok"] is True
    assert source["source_write_api_available"] is False
    assert jobs["items"][0]["kind"] == "scan"
    assert jobs["items"][0]["status"] == "completed"
    assert jobs["next_before_id"] is None
    assert issues["counts"] == {"error": 1}
    assert "relative_path" not in issues["items"][0]
    assert photo["found"] is True
    assert photo["analysis"]["error"] == "decoder failed"
    assert "relative_path" not in photo
    assert collections["items"][0]["decisions"] == {"pending": 1}
    assert audit["read_only"] is True
    assert audit["source_mutation_api_available"] is False
    assert audit["derivative_sample"]["checked"] == 1
    assert audit["derivative_sample"]["missing_examples"] == [
        {"photo_id": 2, "missing": ["thumbnail", "preview"]}
    ]
    assert logs["items"] == []


def test_mcp_bearer_gate_and_diagnostic_sanitizing(settings):
    settings.mcp_bearer_token = SecretStr("mcp-test-secret")
    settings.smb_password = SecretStr("smb-test-secret")
    settings.smb_host = "private-nas.example"
    app = create_app(settings, start_worker=False)

    with TestClient(app) as client:
        response = client.post("/mcp/", json={})
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    sanitizer = DiagnosticSanitizer(settings)
    sanitized = sanitizer.clean(
        "token=mcp-test-secret password=smb-test-secret host=private-nas.example"
    )
    assert "mcp-test-secret" not in sanitized
    assert "smb-test-secret" not in sanitized
    assert "private-nas.example" not in sanitized

    serialized = json.dumps(app.state.settings.model_dump(mode="json"))
    # Pydantic's SecretStr serializer must remain masked if settings are inspected accidentally.
    assert "mcp-test-secret" not in serialized
    assert "smb-test-secret" not in serialized


def test_mcp_host_allowlist_and_disable_switch(settings):
    settings.mcp_allowed_hosts = "photos.local:8787,photos.local:*"
    app = create_app(settings, start_worker=False)
    with TestClient(app) as client:
        rejected = client.post("/mcp/", headers={"host": "other.local:8787"}, json={})
        assert rejected.status_code == 421

    settings.mcp_enabled = False
    disabled_app = create_app(settings, start_worker=False)
    assert disabled_app.state.diagnostic_mcp is None
    with TestClient(disabled_app) as client:
        assert client.post("/mcp/", json={}).status_code == 404

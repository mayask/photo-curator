from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.source import LocalPhotoSource, SMBPhotoSource


def test_local_source_walk_is_filtered_and_read_only(settings, tmp_path: Path):
    nested = settings.photo_root / "trip"
    nested.mkdir()
    payload = b"not really an image, but useful for source testing"
    (nested / "photo.JPG").write_bytes(payload)
    (nested / "notes.txt").write_text("ignore me")
    skipped = settings.photo_root / "@eaDir"
    skipped.mkdir()
    (skipped / "hidden.jpg").write_bytes(b"ignored")

    source = LocalPhotoSource(settings)
    entries = list(source.walk())

    assert [entry.relative_path for entry in entries] == ["trip/photo.JPG"]
    assert entries[0].size_bytes == len(payload)
    assert source.test_connection().startswith("Local photo root is readable")

    destination = tmp_path / "copy.jpg"
    digest, copied = source.copy_to_local_and_hash(
        "trip/photo.JPG", destination, max_bytes=1024
    )
    assert copied == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload


def test_local_source_rejects_path_traversal(settings):
    source = LocalPhotoSource(settings)
    for unsafe_path in ("../secret.jpg", "trip/../../secret.jpg", "/etc/passwd"):
        with pytest.raises(ValueError, match="traversal"):
            with source.open_binary(unsafe_path):
                pass


def test_smb_retry_restores_authenticated_session(settings, monkeypatch):
    calls: list[tuple[str, str]] = []
    fake_client = SimpleNamespace(
        register_session=lambda host, **kwargs: calls.append(("register", host)),
        reset_connection_cache=lambda **kwargs: calls.append(("reset", "")),
    )
    monkeypatch.setitem(sys.modules, "smbclient", fake_client)
    settings.source_mode = "smb"
    settings.smb_host = "nas.test"
    settings.smb_share = "photos"
    settings.smb_user = "reader"
    settings.smb_password = SecretStr("not-logged")
    settings.smb_connection_timeout = 1

    source = SMBPhotoSource(settings)
    attempts = 0

    def transient_operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary disconnect")
        return "ok"

    monkeypatch.setattr("app.source.time.sleep", lambda _delay: None)
    assert source._retry(transient_operation, "test", attempts=2) == "ok"
    assert calls == [
        ("register", "nas.test"),
        ("reset", ""),
        ("register", "nas.test"),
    ]

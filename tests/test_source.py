from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.source import LocalPhotoSource


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

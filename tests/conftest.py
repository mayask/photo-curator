from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.database import reset_database_for_tests


@pytest.fixture(autouse=True)
def clean_database_globals():
    reset_database_for_tests()
    yield
    reset_database_for_tests()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    photo_root = tmp_path / "photos"
    data_dir = tmp_path / "data"
    photo_root.mkdir()
    result = Settings(
        _env_file=None,
        source_mode="local",
        photo_root=photo_root,
        data_dir=data_dir,
        database_url=f"sqlite:///{data_dir / 'test.db'}",
        auto_start=False,
        face_analysis=False,
        scan_interval_hours=0,
        thumb_size=160,
        preview_size=640,
    )
    result.ensure_directories()
    return result

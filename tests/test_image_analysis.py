from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from app.image_analysis import ImageAnalyzer


def test_image_analysis_extracts_metadata_and_builds_derivatives(settings, tmp_path: Path):
    source = tmp_path / "IMG_20240503_141500.jpg"
    height, width = 480, 720
    x = np.linspace(20, 235, width, dtype=np.uint8)
    gradient = np.tile(x, (height, 1))
    rgb = np.dstack((gradient, np.flipud(gradient), np.roll(gradient, 100, axis=1)))
    image = Image.fromarray(rgb, mode="RGB")
    exif = Image.Exif()
    exif[36867] = "2024:05:03 14:15:00"
    exif[271] = "Test Camera Co"
    exif[272] = "Synthetic 1"
    image.save(source, exif=exif, quality=92)

    thumbnail = tmp_path / "cache" / "thumb.jpg"
    preview = tmp_path / "cache" / "preview.jpg"
    result = ImageAnalyzer(settings).analyze(
        source,
        source.stat().st_mtime,
        thumbnail,
        preview,
    )

    assert (result.width, result.height) == (width, height)
    assert result.capture_at == datetime(2024, 5, 3, 14, 15, 0)
    assert result.capture_source == "exif"
    assert result.camera_make == "Test Camera Co"
    assert result.camera_model == "Synthetic 1"
    assert len(result.perceptual_hash or "") == 16
    assert len(result.difference_hash or "") == 16
    assert result.visual_features and len(result.visual_features) == 64 * 4
    assert 0 <= result.quality_score <= 1
    assert thumbnail.is_file()
    assert preview.is_file()
    with Image.open(thumbnail) as thumb_image:
        assert max(thumb_image.size) <= settings.thumb_size
    with Image.open(preview) as preview_image:
        assert max(preview_image.size) <= settings.preview_size

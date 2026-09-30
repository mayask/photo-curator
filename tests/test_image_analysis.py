from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
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


def test_analysis_uses_original_name_for_filename_date(settings, tmp_path: Path):
    local_copy = tmp_path / "temporary-cache-name.jpg"
    Image.new("RGB", (80, 60), "navy").save(local_copy)

    result = ImageAnalyzer(settings).analyze(
        local_copy,
        local_copy.stat().st_mtime,
        tmp_path / "thumb.jpg",
        tmp_path / "preview.jpg",
        source_name="holiday/IMG_19981224_183000.jpg",
    )

    assert result.capture_at == datetime(1998, 12, 24, 18, 30, 0)
    assert result.capture_source == "filename"


def test_analysis_rejects_images_over_pixel_safety_limit(settings, tmp_path: Path):
    source = tmp_path / "small.jpg"
    Image.new("RGB", (80, 60), "navy").save(source)
    settings.max_image_megapixels = 0

    with pytest.raises(ValueError, match="MAX_IMAGE_MEGAPIXELS"):
        ImageAnalyzer(settings).analyze(
            source,
            source.stat().st_mtime,
            tmp_path / "thumb.jpg",
            tmp_path / "preview.jpg",
        )

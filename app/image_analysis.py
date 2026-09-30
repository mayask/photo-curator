from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import ExifTags, Image, ImageOps
from pillow_heif import register_heif_opener

from app.config import Settings

logger = logging.getLogger(__name__)
register_heif_opener()
Image.MAX_IMAGE_PIXELS = 250_000_000


@dataclass(slots=True)
class FaceResult:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    embedding: bytes | None = None


@dataclass(slots=True)
class AnalysisResult:
    width: int
    height: int
    capture_at: datetime
    capture_source: str
    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    iso: int | None = None
    exposure_time: str | None = None
    aperture: float | None = None
    focal_length: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    perceptual_hash: str | None = None
    difference_hash: str | None = None
    visual_features: bytes | None = None
    sharpness_score: float = 0
    exposure_score: float = 0
    contrast_score: float = 0
    color_score: float = 0
    resolution_score: float = 0
    quality_score: float = 0
    faces: list[FaceResult] = field(default_factory=list)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            denominator = float(value.denominator)
            return float(value.numerator) / denominator if denominator else None
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1]) if float(value[1]) else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _clean_text(value: Any, max_length: int = 150) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).replace("\x00", "").strip()
    return text[:max_length] or None


def _gps_decimal(values: Any, reference: Any) -> float | None:
    try:
        degrees = _as_float(values[0])
        minutes = _as_float(values[1])
        seconds = _as_float(values[2])
        if degrees is None or minutes is None or seconds is None:
            return None
        result = degrees + minutes / 60 + seconds / 3600
        ref = _clean_text(reference, 2)
        if ref and ref.upper() in {"S", "W"}:
            result *= -1
        return result
    except (IndexError, TypeError):
        return None


def _extract_gps(exif: Any) -> tuple[float | None, float | None, float | None]:
    gps: dict[Any, Any] = {}
    try:
        if hasattr(ExifTags, "IFD"):
            gps = dict(exif.get_ifd(ExifTags.IFD.GPSInfo))
        if not gps:
            raw = exif.get(34853, {})
            gps = dict(raw) if raw else {}
    except Exception:
        return None, None, None

    latitude = _gps_decimal(gps.get(2), gps.get(1))
    longitude = _gps_decimal(gps.get(4), gps.get(3))
    altitude = _as_float(gps.get(6))
    if gps.get(5) == 1 and altitude is not None:
        altitude *= -1
    if latitude is not None and not -90 <= latitude <= 90:
        latitude = None
    if longitude is not None and not -180 <= longitude <= 180:
        longitude = None
    return latitude, longitude, altitude


def _capture_datetime(exif: Any, path: Path, source_mtime: float) -> tuple[datetime, str]:
    for tag in (36867, 36868, 306):  # DateTimeOriginal, DateTimeDigitized, DateTime
        value = _clean_text(exif.get(tag), 40)
        if not value:
            continue
        normalized = value.split("\x00", 1)[0]
        for format_string in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(normalized[:19], format_string), "exif"
            except ValueError:
                continue

    match = re.search(
        r"(?<!\d)((?:19|20)\d{2})[-_]?([01]\d)[-_]?([0-3]\d)[ T_-]?([0-2]\d)[-_.:]?([0-5]\d)[-_.:]?([0-5]\d)(?!\d)",
        path.stem,
    )
    if match:
        try:
            return datetime(*map(int, match.groups())), "filename"
        except ValueError:
            pass
    return datetime.fromtimestamp(source_mtime, tz=UTC).replace(tzinfo=None), "mtime"


def _exposure_text(value: Any) -> str | None:
    number = _as_float(value)
    if number is None:
        return _clean_text(value, 40)
    if 0 < number < 1:
        denominator = round(1 / number)
        return f"1/{denominator}"
    return f"{number:g}s"


def _feature_histogram(rgb: np.ndarray) -> bytes:
    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.astype(np.float32).reshape(-1).tobytes()


def _quality_metrics(rgb: np.ndarray, width: int, height: int) -> dict[str, float]:
    max_dimension = max(rgb.shape[:2])
    if max_dimension > 1200:
        scale = 1200 / max_dimension
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = float(np.clip(math.log1p(laplacian_variance) / math.log1p(1200), 0, 1))

    luminance = gray.astype(np.float32) / 255.0
    mean_luminance = float(luminance.mean())
    clipped = float(((luminance < 0.015) | (luminance > 0.985)).mean())
    midpoint = max(0.0, 1.0 - abs(mean_luminance - 0.5) / 0.5)
    exposure = float(np.clip(0.7 * midpoint + 0.3 * (1 - clipped), 0, 1))

    contrast_raw = float(luminance.std())
    contrast = float(np.clip(contrast_raw / 0.24, 0, 1))

    rgb_float = rgb.astype(np.float32)
    red, green, blue = cv2.split(rgb_float)
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness_raw = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2) + 0.3 * math.sqrt(
        float(rg.mean()) ** 2 + float(yb.mean()) ** 2
    )
    color = float(np.clip(colorfulness_raw / 90, 0, 1))

    megapixels = width * height / 1_000_000
    resolution = float(np.clip(math.log1p(megapixels) / math.log1p(12), 0, 1))

    return {
        "sharpness": sharpness,
        "exposure": exposure,
        "contrast": contrast,
        "color": color,
        "resolution": resolution,
    }


class ImageTooLargeError(ValueError):
    pass


class FaceAnalyzer:
    def __init__(self, settings: Settings):
        self.enabled = settings.face_analysis
        self.detector: Any | None = None
        self.recognizer: Any | None = None
        self.cascade: Any | None = None
        if not self.enabled:
            return
        try:
            if settings.face_detector_model.is_file():
                self.detector = cv2.FaceDetectorYN.create(
                    str(settings.face_detector_model),
                    "",
                    (320, 320),
                    score_threshold=0.72,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                if settings.face_recognizer_model.is_file():
                    self.recognizer = cv2.FaceRecognizerSF.create(
                        str(settings.face_recognizer_model), ""
                    )
                    logger.info("Using YuNet + SFace for face analysis")
                    return
                logger.info("Using YuNet for face detection (SFace model unavailable)")
                return
        except Exception as exc:
            logger.warning("Could not initialize YuNet/SFace: %s", exc)
            self.detector = None
            self.recognizer = None

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.is_file():
            self.cascade = cv2.CascadeClassifier(str(cascade_path))
            logger.warning("Face models unavailable; using less accurate Haar detection fallback")

    def analyze(self, rgb: np.ndarray) -> list[FaceResult]:
        if not self.enabled:
            return []
        original_height, original_width = rgb.shape[:2]
        scale = min(1.0, 1600 / max(original_width, original_height))
        if scale < 1:
            working = cv2.resize(
                rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            working = rgb
        bgr = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]

        if self.detector is not None:
            try:
                self.detector.setInputSize((width, height))
                _result, detections = self.detector.detect(bgr)
                if detections is None:
                    return []
                faces: list[FaceResult] = []
                for detection in detections:
                    x, y, face_width, face_height = map(float, detection[:4])
                    if face_width < 20 or face_height < 20:
                        continue
                    embedding: bytes | None = None
                    if self.recognizer is not None:
                        try:
                            aligned = self.recognizer.alignCrop(bgr, detection)
                            feature = self.recognizer.feature(aligned).astype(np.float32).reshape(-1)
                            norm = float(np.linalg.norm(feature))
                            if norm:
                                feature /= norm
                                embedding = feature.tobytes()
                        except Exception as exc:
                            logger.debug("Face embedding failed: %s", exc)
                    faces.append(
                        FaceResult(
                            x=max(0.0, x / width),
                            y=max(0.0, y / height),
                            width=min(1.0, face_width / width),
                            height=min(1.0, face_height / height),
                            confidence=float(detection[-1]),
                            embedding=embedding,
                        )
                    )
                return faces
            except Exception as exc:
                logger.warning("YuNet inference failed; falling back for this image: %s", exc)

        if self.cascade is None:
            return []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detections = self.cascade.detectMultiScale(
            gray, scaleFactor=1.12, minNeighbors=5, minSize=(30, 30)
        )
        return [
            FaceResult(
                x=float(x / width),
                y=float(y / height),
                width=float(face_width / width),
                height=float(face_height / height),
                confidence=0.5,
            )
            for x, y, face_width, face_height in detections
        ]


class ImageAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.face_analyzer = FaceAnalyzer(settings)

    def analyze(
        self,
        source_path: Path,
        source_mtime: float,
        thumbnail_path: Path,
        preview_path: Path,
        source_name: str | Path | None = None,
    ) -> AnalysisResult:
        with Image.open(source_path) as original:
            try:
                original.seek(0)
            except EOFError:
                pass
            exif = original.getexif()
            logical_path = Path(source_name) if source_name is not None else source_path
            capture_at, capture_source = _capture_datetime(
                exif, logical_path, source_mtime
            )
            latitude, longitude, altitude = _extract_gps(exif)

            raw_width, raw_height = original.size
            megapixels = raw_width * raw_height / 1_000_000
            if megapixels > self.settings.max_image_megapixels:
                raise ImageTooLargeError(
                    f"Image has {megapixels:.1f} megapixels; limit is "
                    f"MAX_IMAGE_MEGAPIXELS={self.settings.max_image_megapixels}"
                )
            orientation = int(_as_float(exif.get(274)) or 1)
            if orientation in {5, 6, 7, 8}:
                width, height = raw_height, raw_width
            else:
                width, height = raw_width, raw_height

            # JPEG decoders can select a lower-resolution DCT level before loading.
            # All current analysis operates at preview scale, while the original
            # dimensions above are retained for print-resolution scoring.
            working_limit = max(1600, self.settings.preview_size, self.settings.thumb_size)
            original.draft("RGB", (working_limit, working_limit))
            oriented = ImageOps.exif_transpose(original)
            if oriented.mode not in {"RGB", "RGBA"}:
                oriented = oriented.convert("RGB")
            elif oriented.mode == "RGBA":
                background = Image.new("RGB", oriented.size, "white")
                background.paste(oriented, mask=oriented.getchannel("A"))
                oriented = background
            else:
                oriented = oriented.copy()
            oriented.thumbnail(
                (working_limit, working_limit), Image.Resampling.LANCZOS
            )

        rgb = np.asarray(oriented, dtype=np.uint8)
        metrics = _quality_metrics(rgb, width, height)
        faces = self.face_analyzer.analyze(rgb)

        face_bonus = 0.0
        if faces:
            largest_face = max(face.width * face.height for face in faces)
            face_bonus = min(0.08, 0.025 + largest_face * 0.25)
        quality = float(
            np.clip(
                metrics["sharpness"] * 0.30
                + metrics["exposure"] * 0.24
                + metrics["contrast"] * 0.14
                + metrics["color"] * 0.12
                + metrics["resolution"] * 0.20
                + face_bonus,
                0,
                1,
            )
        )

        self._save_derivative(oriented, thumbnail_path, self.settings.thumb_size)
        self._save_derivative(oriented, preview_path, self.settings.preview_size)

        return AnalysisResult(
            width=width,
            height=height,
            capture_at=capture_at,
            capture_source=capture_source,
            camera_make=_clean_text(exif.get(271), 100),
            camera_model=_clean_text(exif.get(272), 100),
            lens_model=_clean_text(exif.get(42036), 150),
            iso=int(_as_float(exif.get(34855))) if _as_float(exif.get(34855)) is not None else None,
            exposure_time=_exposure_text(exif.get(33434)),
            aperture=_as_float(exif.get(33437)),
            focal_length=_as_float(exif.get(37386)),
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            perceptual_hash=str(imagehash.phash(oriented, hash_size=8)),
            difference_hash=str(imagehash.dhash(oriented, hash_size=8)),
            visual_features=_feature_histogram(rgb),
            sharpness_score=metrics["sharpness"],
            exposure_score=metrics["exposure"],
            contrast_score=metrics["contrast"],
            color_score=metrics["color"],
            resolution_score=metrics["resolution"],
            quality_score=quality,
            faces=faces,
        )

    def _save_derivative(self, image: Image.Image, path: Path, maximum_size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        derivative = image.copy()
        derivative.thumbnail((maximum_size, maximum_size), Image.Resampling.LANCZOS)
        temporary = path.with_name(path.name + ".tmp")
        derivative.save(
            temporary,
            format="JPEG",
            quality=self.settings.jpeg_quality,
            optimize=True,
            progressive=True,
        )
        temporary.replace(path)


class PlaceResolver:
    """Offline nearest-city lookup with rounded-coordinate caching."""

    def __init__(self) -> None:
        self._cache: dict[tuple[float, float], tuple[str | None, str | None, str | None]] = {}
        self._module: Any | None = None
        self._failed = False

    def resolve(
        self, latitude: float | None, longitude: float | None
    ) -> tuple[str | None, str | None, str | None]:
        if latitude is None or longitude is None:
            return None, None, None
        key = (round(latitude, 2), round(longitude, 2))
        if key in self._cache:
            return self._cache[key]
        if self._failed:
            return None, None, None
        try:
            if self._module is None:
                import reverse_geocoder

                self._module = reverse_geocoder
            result = self._module.search((latitude, longitude), mode=1)[0]
            value = (
                _clean_text(result.get("name"), 120),
                _clean_text(result.get("admin1"), 120),
                _clean_text(result.get("cc"), 120),
            )
            self._cache[key] = value
            return value
        except Exception as exc:
            logger.warning("Offline place lookup unavailable: %s", exc)
            self._failed = True
            return None, None, None

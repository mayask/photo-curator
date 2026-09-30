from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Photo Book Curator"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/photo-curator.db"

    source_mode: Literal["smb", "local"] = "smb"
    photo_root: Path = Path("/photos")
    smb_host: str = ""
    smb_share: str = ""
    smb_path: str = ""
    smb_user: str = ""
    smb_password: SecretStr = SecretStr("")
    smb_domain: str = ""
    smb_port: int = Field(default=445, ge=1, le=65535)
    smb_connection_timeout: int = Field(default=30, ge=1, le=600)

    auto_start: bool = True
    scan_interval_hours: float = Field(default=24.0, ge=0)
    worker_batch_size: int = Field(default=25, ge=1, le=10_000)
    worker_poll_seconds: float = Field(default=2.0, ge=0.1, le=60)
    max_analysis_attempts: int = Field(default=3, ge=1, le=100)
    max_file_mb: int = Field(default=250, ge=1)
    max_image_megapixels: int = Field(default=100, ge=1)

    thumb_size: int = Field(default=360, ge=64, le=2000)
    preview_size: int = Field(default=1600, ge=320, le=8000)
    jpeg_quality: int = Field(default=84, ge=25, le=100)
    face_analysis: bool = True
    face_detector_model: Path = Path("/app/models/face_detection_yunet_2023mar.onnx")
    face_recognizer_model: Path = Path(
        "/app/models/face_recognition_sface_2021dec.onnx"
    )

    holiday_country: str = ""
    holiday_subdiv: str = ""
    event_gap_hours: float = Field(default=8.0, gt=0, le=168)
    burst_gap_seconds: int = Field(default=45, ge=1, le=3600)
    gps_cluster_km: float = Field(default=25.0, gt=0, le=1000)

    skip_directories: str = "@eaDir,#recycle,.snapshot,.thumbnails"
    supported_extensions: str = (
        ".jpg,.jpeg,.jpe,.png,.webp,.heic,.heif,.tif,.tiff,.bmp,.gif"
    )

    analysis_version: int = 1
    collection_version: int = 1

    @field_validator("smb_path")
    @classmethod
    def normalize_smb_path(cls, value: str) -> str:
        return value.strip().strip("/\\")

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def thumbs_dir(self) -> Path:
        return self.cache_dir / "thumbs"

    @property
    def previews_dir(self) -> Path:
        return self.cache_dir / "previews"

    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def extension_set(self) -> frozenset[str]:
        return frozenset(
            item.strip().lower()
            if item.strip().startswith(".")
            else f".{item.strip().lower()}"
            for item in self.supported_extensions.split(",")
            if item.strip()
        )

    @property
    def skipped_directory_set(self) -> frozenset[str]:
        return frozenset(item.strip().casefold() for item in self.skip_directories.split(",") if item.strip())

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.cache_dir,
            self.thumbs_dir,
            self.previews_dir,
            self.temp_dir,
            self.export_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_source(self) -> None:
        if self.source_mode == "local":
            if not self.photo_root.exists() or not self.photo_root.is_dir():
                raise ValueError(f"PHOTO_ROOT does not exist or is not a directory: {self.photo_root}")
            return
        missing = [
            name
            for name, value in (
                ("SMB_HOST", self.smb_host),
                ("SMB_SHARE", self.smb_share),
                ("SMB_USER", self.smb_user),
                ("SMB_PASSWORD", self.smb_password.get_secret_value()),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing SMB settings: {', '.join(missing)}")

    def source_summary(self) -> str:
        if self.source_mode == "local":
            return f"local:{self.photo_root} (read-only expected)"
        suffix = f"/{self.smb_path}" if self.smb_path else ""
        return f"smb://{self.smb_host}/{self.smb_share}{suffix}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

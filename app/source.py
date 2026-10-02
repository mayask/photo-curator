from __future__ import annotations

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceEntry:
    relative_path: str
    name: str
    size_bytes: int
    mtime: float


class SourceError(RuntimeError):
    pass


class SourceSizeLimitError(SourceError):
    pass


def _safe_relative_path(relative_path: str) -> str:
    candidate = PurePosixPath(relative_path.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Path traversal outside the configured photo root is not allowed")
    return "/".join(part for part in candidate.parts if part not in ("", "."))


class PhotoSource(ABC):
    """Read-only interface to a photo tree.

    Deliberately no write/delete methods are exposed. Both implementations only open
    source files in binary read mode.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.walk_warnings = 0

    @abstractmethod
    def walk(self) -> Iterator[SourceEntry]:
        raise NotImplementedError

    @abstractmethod
    @contextmanager
    def open_binary(self, relative_path: str) -> Generator[BinaryIO, None, None]:
        raise NotImplementedError

    @abstractmethod
    def test_connection(self) -> str:
        raise NotImplementedError

    def copy_to_local_and_hash(
        self,
        relative_path: str,
        destination: Path,
        max_bytes: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[str, int]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        total = 0
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with self.open_binary(relative_path) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise SourceSizeLimitError(
                            f"File exceeds MAX_FILE_MB safety limit ({max_bytes} bytes)"
                        )
                    digest.update(chunk)
                    target.write(chunk)
                    if progress:
                        progress(total)
            temporary.replace(destination)
            return digest.hexdigest(), total
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


class LocalPhotoSource(PhotoSource):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.root = settings.photo_root.expanduser().resolve()

    def walk(self) -> Iterator[SourceEntry]:
        skipped = self.settings.skipped_directory_set
        stack = [self.root]
        while stack:
            directory = stack.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError as exc:
                self.walk_warnings += 1
                logger.warning("Cannot list local directory %s: %s", directory, exc)
                continue
            entries.sort(key=lambda item: item.name.casefold(), reverse=True)
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.casefold() not in skipped:
                            stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    extension = Path(entry.name).suffix.lower()
                    if extension not in self.settings.extension_set:
                        continue
                    stat = entry.stat(follow_symlinks=False)
                    path = Path(entry.path).resolve()
                    try:
                        relative = path.relative_to(self.root).as_posix()
                    except ValueError:
                        continue
                    yield SourceEntry(relative, entry.name, stat.st_size, stat.st_mtime)
                except OSError as exc:
                    self.walk_warnings += 1
                    logger.warning("Skipping local entry %s: %s", entry.path, exc)

    @contextmanager
    def open_binary(self, relative_path: str) -> Generator[BinaryIO, None, None]:
        safe_path = _safe_relative_path(relative_path)
        path = (self.root / safe_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise SourceError("Resolved path is outside PHOTO_ROOT") from exc
        with path.open("rb") as handle:
            yield handle

    def test_connection(self) -> str:
        if not self.root.is_dir():
            raise SourceError(f"PHOTO_ROOT is not a directory: {self.root}")
        with os.scandir(self.root) as entries:
            next(entries, None)
        return f"Local photo root is readable: {self.root}"


class SMBPhotoSource(PhotoSource):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        try:
            import smbclient
        except ImportError as exc:  # pragma: no cover - dependency is in the image
            raise SourceError("smbprotocol is required for SOURCE_MODE=smb") from exc
        self.smbclient = smbclient
        self._host = settings.smb_host
        self._username = settings.smb_user
        if settings.smb_domain:
            self._username = f"{settings.smb_domain}\\{self._username}"
        self._password = settings.smb_password.get_secret_value()
        self._port = settings.smb_port
        self._connection_timeout = settings.smb_connection_timeout
        self._register_session()
        root_parts = [settings.smb_host, settings.smb_share]
        if settings.smb_path:
            root_parts.extend(
                part
                for part in settings.smb_path.replace("/", "\\").split("\\")
                if part
            )
        self.root = "\\\\" + "\\".join(root_parts)

    def _register_session(self) -> None:
        self.smbclient.register_session(
            self._host,
            username=self._username,
            password=self._password,
            port=self._port,
            connection_timeout=self._connection_timeout,
        )

    def _remote_path(self, relative_path: str = "") -> str:
        safe_path = _safe_relative_path(relative_path)
        if not safe_path:
            return self.root
        return self.root + "\\" + safe_path.replace("/", "\\")

    def _retry(self, operation: Callable[[], Any], label: str, attempts: int = 4) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except SourceSizeLimitError:
                raise
            except Exception as exc:  # SMB library exposes several transport errors
                last_error = exc
                if attempt == attempts:
                    break
                delay = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "SMB %s failed (attempt %d/%d); retrying in %ss: %s",
                    label,
                    attempt,
                    attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
                try:
                    self.smbclient.reset_connection_cache(fail_on_error=False)
                except Exception:
                    pass
                # reset_connection_cache also drops the authenticated session.
                # Re-register it before retrying or every later read falls back to
                # unauthenticated negotiation despite valid configured credentials.
                try:
                    self._register_session()
                except Exception as reconnect_error:
                    logger.warning("SMB session re-registration failed: %s", reconnect_error)
        raise SourceError(f"SMB {label} failed after {attempts} attempts: {last_error}")

    def walk(self) -> Iterator[SourceEntry]:
        skipped = self.settings.skipped_directory_set
        stack: list[tuple[str, str]] = [("", self.root)]
        while stack:
            relative_directory, remote_directory = stack.pop()

            def scan(path: str = remote_directory) -> list[Any]:
                with self.smbclient.scandir(path) as iterator:
                    return list(iterator)

            try:
                entries = self._retry(scan, f"listing {relative_directory or '/'}")
            except SourceError as exc:
                self.walk_warnings += 1
                logger.error("Skipping unreadable SMB directory %s: %s", relative_directory, exc)
                continue
            entries.sort(key=lambda item: item.name.casefold(), reverse=True)
            for entry in entries:
                name = entry.name
                if name in {".", ".."}:
                    continue
                relative = f"{relative_directory}/{name}" if relative_directory else name
                relative = relative.replace("\\", "/")
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if name.casefold() not in skipped:
                            stack.append((relative, self._remote_path(relative)))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if Path(name).suffix.lower() not in self.settings.extension_set:
                        continue
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except TypeError:
                        stat = entry.stat()
                    yield SourceEntry(relative, name, int(stat.st_size), float(stat.st_mtime))
                except Exception as exc:
                    self.walk_warnings += 1
                    logger.warning("Skipping SMB entry %s: %s", relative, exc)

    @contextmanager
    def open_binary(self, relative_path: str) -> Generator[BinaryIO, None, None]:
        remote_path = self._remote_path(relative_path)
        handle = self._retry(
            lambda: self.smbclient.open_file(
                remote_path,
                mode="rb",
                buffering=1024 * 1024,
            ),
            f"opening {relative_path}",
        )
        try:
            yield handle
        finally:
            handle.close()

    def copy_to_local_and_hash(
        self,
        relative_path: str,
        destination: Path,
        max_bytes: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[str, int]:
        def copy_once() -> tuple[str, int]:
            return super(SMBPhotoSource, self).copy_to_local_and_hash(
                relative_path,
                destination,
                max_bytes,
                progress,
            )

        return self._retry(copy_once, f"reading {relative_path}", attempts=3)

    def test_connection(self) -> str:
        def probe() -> None:
            with self.smbclient.scandir(self.root) as iterator:
                next(iterator, None)

        self._retry(probe, "connection test", attempts=2)
        return "SMB share and configured path are readable"


def build_source(settings: Settings) -> PhotoSource:
    settings.validate_source()
    if settings.source_mode == "local":
        return LocalPhotoSource(settings)
    return SMBPhotoSource(settings)

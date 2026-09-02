"""Durable JSON Lines recording for run events."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prefscope.observability.events import RunEvent

DEFAULT_MAX_EVENT_BYTES = 1_048_576


def _required_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if value is None:
        raise RuntimeError(f"secure JSONL recording requires os.{name}")
    return int(value)


def _open_parent_directory(path: Path) -> int:
    """Create and open a parent path without following any component symlink."""
    absolute = Path(os.path.abspath(path))
    flags = (
        os.O_RDONLY
        | _required_open_flag("O_DIRECTORY")
        | _required_open_flag("O_NOFOLLOW")
        | _required_open_flag("O_CLOEXEC")
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    # Another creator won the race. The no-follow open below
                    # still verifies that the new component is a directory.
                    pass
                else:
                    os.fsync(descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ValueError(
                    f"JSONL trace parent contains an unsafe component: {path}"
                ) from exc
            os.close(descriptor)
            descriptor = child
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


class JsonlRecorder:
    """Append validated run events to a securely opened JSONL file.

    Each call to :meth:`record` writes one bounded UTF-8 line and, by default,
    calls ``fsync``. The parent directory is created explicitly and synchronized
    when the trace file is new. Final-component symlinks and all non-regular
    files are rejected. File permissions are forced to ``0600``.

    Writes are thread-safe within one recorder. Coordinate separate recorders or
    processes externally if they share a path.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        run_id: str | None = None,
        durable: bool = True,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(max_event_bytes, bool) or not isinstance(max_event_bytes, int):
            raise ValueError("max_event_bytes must be a positive integer")
        if max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be a positive integer")

        self.path = Path(path)
        self.run_id = run_id or str(uuid.uuid4())
        self.durable = durable
        self.max_event_bytes = max_event_bytes
        self._monotonic = monotonic
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._started_at = self._monotonic()
        self._lock = threading.RLock()
        self._closed = False
        self._descriptor = self._open_securely()

    def _open_securely(self) -> int:
        try:
            parent_descriptor = _open_parent_directory(self.path.parent)
        except OSError as exc:
            raise ValueError(
                f"could not securely create JSONL trace parent: {self.path.parent}"
            ) from exc

        filename = self.path.name
        try:
            try:
                path_info = os.stat(
                    filename,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                path_info = None

            if path_info is not None:
                if stat.S_ISLNK(path_info.st_mode):
                    raise ValueError(
                        f"JSONL trace path must not be a symlink: {self.path}"
                    )
                if not stat.S_ISREG(path_info.st_mode):
                    raise ValueError(
                        f"JSONL trace path must be a regular file: {self.path}"
                    )

            flags = (
                os.O_WRONLY
                | os.O_APPEND
                | getattr(os, "O_NONBLOCK", 0)
                | _required_open_flag("O_NOFOLLOW")
                | _required_open_flag("O_CLOEXEC")
            )
            created = path_info is None
            if created:
                flags |= os.O_CREAT | os.O_EXCL

            try:
                descriptor = os.open(
                    filename,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError as exc:
                raise ValueError(
                    f"JSONL trace path changed while it was being opened: {self.path}"
                ) from exc
            except OSError as exc:
                raise ValueError(
                    f"could not securely open JSONL trace path: {self.path}"
                ) from exc

            try:
                opened_info = os.fstat(descriptor)
                if not stat.S_ISREG(opened_info.st_mode):
                    raise ValueError(
                        f"JSONL trace path must be a regular file: {self.path}"
                    )
                opened_identity = (opened_info.st_dev, opened_info.st_ino)
                if path_info is not None and opened_identity != (
                    path_info.st_dev,
                    path_info.st_ino,
                ):
                    raise ValueError(
                        f"JSONL trace path changed while opening: {self.path}"
                    )
                entry_info = os.stat(
                    filename,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if opened_identity != (entry_info.st_dev, entry_info.st_ino):
                    raise ValueError(
                        f"JSONL trace directory entry changed while opening: {self.path}"
                    )
                os.fchmod(descriptor, 0o600)
                if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
                    raise ValueError(
                        f"JSONL trace permissions must be 0600: {self.path}"
                    )
                if created:
                    os.fsync(parent_descriptor)
            except Exception:
                os.close(descriptor)
                raise
            return descriptor
        finally:
            os.close(parent_descriptor)

    @property
    def closed(self) -> bool:
        return self._closed

    def record(
        self,
        stage: str,
        status: str,
        message: str = "",
        data: Mapping[str, Any] | None = None,
    ) -> RunEvent:
        """Validate and durably append one bounded event."""
        with self._lock:
            if self._closed:
                raise ValueError("cannot record to a closed JsonlRecorder")
            now = self._utc_now()
            if not isinstance(now, datetime) or now.tzinfo is None:
                raise ValueError("utc_now must return a timezone-aware datetime")
            timestamp = (
                now.astimezone(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            elapsed = self._monotonic() - self._started_at
            event = RunEvent(
                run_id=self.run_id,
                timestamp=timestamp,
                elapsed_seconds=elapsed,
                stage=stage,
                status=status,
                message=message,
                data={} if data is None else data,
            )
            payload = (
                json.dumps(
                    event.to_dict(),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            if len(payload) > self.max_event_bytes:
                raise ValueError(
                    f"serialized event exceeds maximum size {self.max_event_bytes} bytes"
                )
            self._write_all(payload)
            self.flush()
            return event

    def _write_all(self, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(self._descriptor, view)
            if written <= 0:
                raise OSError("could not append JSONL event")
            view = view[written:]

    def record_failure(
        self,
        stage: str,
        error: BaseException,
        *,
        message: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> RunEvent:
        """Record an exception without storing a traceback or exception arguments."""
        failure_data = dict(data or {})
        failure_data["error_type"] = type(error).__name__
        failure_data["error_message"] = str(error)
        return self.record(
            stage,
            "failed",
            message if message is not None else f"{type(error).__name__}: {error}",
            failure_data,
        )

    def flush(self) -> None:
        """Synchronize the descriptor to storage when durability is enabled."""
        with self._lock:
            if self._closed or not self.durable:
                return
            os.fsync(self._descriptor)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self.flush()
            finally:
                os.close(self._descriptor)
                self._closed = True

    def __enter__(self) -> JsonlRecorder:
        if self._closed:
            raise ValueError("cannot enter a closed JsonlRecorder")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

"""Adapters from Python logging and warnings to run events."""

from __future__ import annotations

import copy
import logging
import threading
import warnings
from pathlib import Path
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from prefscope.observability.recorder import JsonlRecorder


class RecorderLoggingHandler(logging.Handler):
    """Write standard-library log records to a :class:`JsonlRecorder`."""

    def __init__(
        self,
        recorder: JsonlRecorder,
        *,
        stage: str = "logging",
        level: int = logging.NOTSET,
    ) -> None:
        super().__init__(level=level)
        self.recorder = recorder
        self.stage = stage

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= logging.ERROR:
                status = "failed"
            elif record.levelno >= logging.WARNING:
                status = "warning"
            else:
                status = "info"
            data: dict[str, Any] = {
                "logger": record.name,
                "level": record.levelname,
            }
            if record.exc_info and record.exc_info[0] is not None:
                data["error_type"] = record.exc_info[0].__name__
            # Format a copy without traceback or stack text. Those can contain
            # source lines and local paths that are not safe observability data.
            safe_record = copy.copy(record)
            safe_record.exc_info = None
            safe_record.exc_text = None
            safe_record.stack_info = None
            self.recorder.record(
                self.stage,
                status,
                self.format(safe_record),
                data,
            )
        except Exception:
            self.handleError(record)


_WARNING_BRIDGE_LOCK = threading.RLock()


@contextmanager
def capture_warnings(
    recorder: JsonlRecorder,
    *,
    stage: str = "warnings",
    forward: bool = True,
) -> Iterator[None]:
    """Record warnings emitted in the context using the standard warnings hook.

    The Python warnings hook is process-global. Warnings from any thread during
    the context are attributed to this recorder, so do not overlap bridges for
    separate runs. Existing warning filters still decide which warnings are
    emitted. Recording is best effort and normal warning output is preserved by
    default.
    """
    with _WARNING_BRIDGE_LOCK:
        previous = warnings.showwarning

        def showwarning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            try:
                recorder.record(
                    stage,
                    "warning",
                    str(message),
                    {
                        "category": category.__name__,
                        "filename": Path(filename).name,
                        "lineno": lineno,
                    },
                )
            except Exception:
                # Observability must not turn a non-fatal warning into a failure.
                pass
            if forward:
                previous(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = showwarning
        try:
            yield
        finally:
            warnings.showwarning = previous

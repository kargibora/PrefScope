"""Compact, privacy-safe console reporting for run events."""

from __future__ import annotations

import builtins
import math
import sys
import threading
from collections.abc import Mapping
from typing import Any, TextIO

from prefscope.observability.events import RunEvent

DEFAULT_MAX_CONSOLE_LINES = 200
_MAX_DISPLAY_COUNT = 10**15
_MAX_DISPLAY_DURATION_SECONDS = 10**9
_OUTPUT_LOCK = threading.RLock()

# These labels are code-owned. Event stage names are never rendered directly.
_STAGE_LABELS = {
    "analysis_result.load": "Load analysis result",
    "analysis_result.save": "Save analysis result",
    "analyze_dataset": "Analyze dataset",
    "analyze_preference": "Analyze preference",
    "encode": "Encode",
    "encode_pairs": "Encode pairs",
    "feature_bundle.load": "Load feature bundle",
    "feature_bundle.save": "Save feature bundle",
    "featurize": "Featurize",
    "fetch_lens": "Fetch lens",
    "load_feature_source": "Load feature source",
    "load_lens": "Load lens",
    "project_representations": "Project representations",
    "report_bundle.load": "Load report bundle",
    "report_bundle.write": "Write report bundle",
    "save_lens": "Save lens",
}

_COUNT_FIELDS = (
    (("output_rows", "n_rows", "rows", "input_rows"), "row", "rows"),
    (("output_features", "n_features", "feature_width"), "feature", "features"),
    (("n_groups",), "group", "groups"),
    (("n_views", "feature_view_count"), "view", "views"),
    (("n_arrays",), "array", "arrays"),
    (("artifact_count",), "artifact", "artifacts"),
    (("evidence_layer_count",), "evidence layer", "evidence layers"),
)
_SAFE_STATUSES = frozenset({"ready", "partial", "failed", "unavailable", "error"})
_SAFE_PROFILES = frozenset({"local", "shareable"})
_STATUS_STYLES = {
    "completed": "green",
    "failed": "red",
    "warning": "yellow",
}
_SUPPRESSION_STYLE = "dim"


def _safe_builtin_name(value: object, base: type[BaseException], fallback: str) -> str:
    if type(value) is not str:
        return fallback
    resolved = getattr(builtins, value, None)
    if (
        not isinstance(resolved, type)
        or not issubclass(resolved, base)
        or resolved.__name__ != value
    ):
        return fallback
    return resolved.__name__


def _safe_error_name(data: Mapping[str, Any]) -> str:
    return _safe_builtin_name(data.get("error_type"), BaseException, "Exception")


def _safe_warning_name(data: Mapping[str, Any]) -> str:
    return _safe_builtin_name(data.get("category"), Warning, "Warning")


def _non_negative_count(value: object) -> int | None:
    if type(value) is not int or value < 0 or value > _MAX_DISPLAY_COUNT:
        return None
    return value


def _details(data: Mapping[str, Any]) -> list[str]:
    details: list[str] = []
    duration = data.get("duration_seconds")
    if (
        type(duration) in {int, float}
        and math.isfinite(float(duration))
        and 0 <= float(duration) <= _MAX_DISPLAY_DURATION_SECONDS
    ):
        details.append(f"{float(duration):.2f}s")

    for keys, singular, plural in _COUNT_FIELDS:
        count = None
        for key in keys:
            count = _non_negative_count(data.get(key))
            if count is not None:
                break
        if count is not None:
            unit = singular if count == 1 else plural
            details.append(f"{count:,} {unit}")

    status = data.get("status")
    if type(status) is str and status in _SAFE_STATUSES:
        details.append(status)
    profile = data.get("profile")
    if type(profile) is str and profile in _SAFE_PROFILES:
        details.append(f"{profile} profile")
    return details


def _render_event(event: RunEvent) -> str | None:
    """Render only code-owned labels and allowlisted scalar details."""
    if event.status in {"started", "info"}:
        return None
    data = event.data
    if event.status == "warning":
        return f"⚠ {_safe_warning_name(data)}"
    if event.stage == "logging" and event.status == "failed":
        return f"✗ Error  {_safe_error_name(data)}"

    label = _STAGE_LABELS.get(event.stage, "Operation")
    if event.status not in {"completed", "failed"}:
        return None

    marker = "✓" if event.status == "completed" else "✗"
    details = _details(data)
    if event.status == "failed":
        details.append(_safe_error_name(data))
    suffix = f"  {' · '.join(details)}" if details else ""
    return f"{marker} {label}{suffix}"


class ConsoleReporter:
    """Write a bounded stream of compact event summaries to stderr.

    Rich is optional and imported only when a reporter is constructed. If it
    cannot be imported or initialized, the reporter writes plain text.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        max_lines: int = DEFAULT_MAX_CONSOLE_LINES,
        _force_terminal: bool | None = None,
    ) -> None:
        if (
            isinstance(max_lines, bool)
            or not isinstance(max_lines, int)
            or max_lines < 0
        ):
            raise ValueError("max_lines must be a non-negative integer")
        self._stream = sys.stderr if stream is None else stream
        self._max_lines = max_lines
        self._line_count = 0
        self._suppressed = False
        self._closed = False
        self._lock = _OUTPUT_LOCK
        self._console: Any | None = None
        try:
            from rich.console import Console

            if _force_terminal is None:
                isatty = getattr(self._stream, "isatty", None)
                _force_terminal = bool(isatty()) if isatty is not None else False
            self._console = Console(
                file=self._stream,
                force_terminal=_force_terminal,
                highlight=False,
                markup=False,
                soft_wrap=True,
            )
        except BaseException:
            self._console = None

    def _write(self, line: str, *, style: str) -> None:
        if self._console is not None:
            self._console.print(line, style=style)
        else:
            self._stream.write(f"{line}\n")
            self._stream.flush()

    def observe(self, event: RunEvent) -> None:
        """Print a terminal event, subject to the output bound."""
        line = _render_event(event)
        if line is None:
            return
        with self._lock:
            if self._closed:
                return
            if self._line_count < self._max_lines:
                self._write(line, style=_STATUS_STYLES[event.status])
                self._line_count += 1
                return
            if not self._suppressed:
                self._write(
                    "… Further observability output suppressed",
                    style=_SUPPRESSION_STYLE,
                )
                self._suppressed = True

    def close(self) -> None:
        """Stop accepting output. Repeated calls are safe."""
        with self._lock:
            self._closed = True


__all__ = ["ConsoleReporter", "DEFAULT_MAX_CONSOLE_LINES"]

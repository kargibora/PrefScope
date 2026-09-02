"""Torch-free run observability primitives."""

from prefscope.core.redaction import REDACTED
from prefscope.observability.events import (
    DEFAULT_MAX_EVENT_NODES,
    DEFAULT_MAX_NESTING_DEPTH,
    DEFAULT_MAX_STRING_LENGTH,
    EVENT_SCHEMA_VERSION,
    RUN_STATUSES,
    RunEvent,
)
from prefscope.observability.handlers import RecorderLoggingHandler, capture_warnings
from prefscope.observability.recorder import DEFAULT_MAX_EVENT_BYTES, JsonlRecorder
from prefscope.observability.runtime import RunContext, observe_run

__all__ = [
    "DEFAULT_MAX_EVENT_BYTES",
    "DEFAULT_MAX_EVENT_NODES",
    "DEFAULT_MAX_NESTING_DEPTH",
    "DEFAULT_MAX_STRING_LENGTH",
    "EVENT_SCHEMA_VERSION",
    "REDACTED",
    "RUN_STATUSES",
    "JsonlRecorder",
    "RecorderLoggingHandler",
    "RunContext",
    "RunEvent",
    "capture_warnings",
    "observe_run",
]

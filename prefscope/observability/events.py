"""Portable event schema for PrefScope run observability."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar

from prefscope.core.redaction import redact_secrets, redact_text

EVENT_SCHEMA_VERSION = 1
RUN_STATUSES = frozenset({"started", "completed", "info", "warning", "failed"})
DEFAULT_MAX_NESTING_DEPTH = 20
DEFAULT_MAX_EVENT_NODES = 10_000
DEFAULT_MAX_STRING_LENGTH = 100_000

_UTC_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


def _check_string(value: str, *, where: str) -> None:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{where} must be valid UTF-8") from exc
    if len(value) > DEFAULT_MAX_STRING_LENGTH:
        raise ValueError(
            f"{where} exceeds maximum string length {DEFAULT_MAX_STRING_LENGTH}"
        )


def _validate_json_limits(
    value: Any,
    *,
    depth: int,
    nodes: list[int],
    ancestors: set[int],
) -> Any:
    nodes[0] += 1
    if nodes[0] > DEFAULT_MAX_EVENT_NODES:
        raise ValueError(
            f"event data exceeds maximum node count {DEFAULT_MAX_EVENT_NODES}"
        )
    if depth > DEFAULT_MAX_NESTING_DEPTH:
        raise ValueError(
            f"event data exceeds maximum nesting depth {DEFAULT_MAX_NESTING_DEPTH}"
        )
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        _check_string(value, where="event data string")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("event data cannot contain NaN or infinity")
        return value

    value_id = id(value)
    if value_id in ancestors:
        raise ValueError("event data cannot contain cycles")
    if isinstance(value, Mapping):
        ancestors.add(value_id)
        output: dict[str, Any] = {}
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("event data object keys must be strings")
                _check_string(key, where="event data key")
                output[key] = _validate_json_limits(
                    item,
                    depth=depth + 1,
                    nodes=nodes,
                    ancestors=ancestors,
                )
        finally:
            ancestors.remove(value_id)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ancestors.add(value_id)
        try:
            return [
                _validate_json_limits(
                    item,
                    depth=depth + 1,
                    nodes=nodes,
                    ancestors=ancestors,
                )
                for item in value
            ]
        finally:
            ancestors.remove(value_id)
    raise ValueError(
        "event data values must be JSON-compatible: null, bool, number, string, array, or object"
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sanitize_json(value: Mapping[str, Any]) -> Mapping[str, Any]:
    bounded_copy = _validate_json_limits(value, depth=0, nodes=[0], ancestors=set())
    clean = redact_secrets(bounded_copy, where="event data")
    return _freeze_json(clean)


def _to_mutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_mutable_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_mutable_json(item) for item in value]
    return value


def _validate_timestamp(value: str) -> None:
    if not value.endswith("Z"):
        raise ValueError("event timestamp must be UTC and end in 'Z'")
    if _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError("event timestamp must use canonical ISO 8601 UTC syntax")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError("event timestamp must be an ISO 8601 UTC timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("event timestamp must be UTC")


@dataclass(frozen=True, slots=True)
class RunEvent:
    """An immutable, versioned record of one run observation."""

    run_id: str
    timestamp: str
    elapsed_seconds: float
    stage: str
    status: str
    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = EVENT_SCHEMA_VERSION

    VERSION: ClassVar[int] = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != EVENT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported event schema_version {self.schema_version}; expected {EVENT_SCHEMA_VERSION}"
            )
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("event run_id must be a non-empty string")
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("event stage must be a non-empty string")
        if not isinstance(self.status, str) or self.status not in RUN_STATUSES:
            allowed = ", ".join(sorted(RUN_STATUSES))
            raise ValueError(f"event status must be one of: {allowed}")
        if isinstance(self.elapsed_seconds, bool) or not isinstance(
            self.elapsed_seconds, (int, float)
        ):
            raise ValueError("event elapsed_seconds must be a number")
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("event elapsed_seconds must be finite and non-negative")
        if not isinstance(self.timestamp, str):
            raise ValueError("event timestamp must be a string")
        _validate_timestamp(self.timestamp)
        if not isinstance(self.message, str):
            raise ValueError("event message must be a string")
        if not isinstance(self.data, Mapping):
            raise ValueError("event data must be an object")

        for value, where in (
            (self.run_id, "event run_id"),
            (self.stage, "event stage"),
            (self.message, "event message"),
        ):
            _check_string(value, where=where)
        object.__setattr__(self, "run_id", redact_text(self.run_id))
        object.__setattr__(self, "stage", redact_text(self.stage))
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "message", redact_text(self.message))
        object.__setattr__(self, "data", _sanitize_json(self.data))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
            "data": _to_mutable_json(self.data),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunEvent:
        """Parse and validate a serialized event, rejecting unknown fields."""
        if not isinstance(value, Mapping):
            raise ValueError("serialized event must be an object")
        if not all(isinstance(key, str) for key in value):
            raise ValueError("serialized event field names must be strings")
        expected = {
            "schema_version",
            "run_id",
            "timestamp",
            "elapsed_seconds",
            "stage",
            "status",
            "message",
            "data",
        }
        missing = expected.difference(value)
        extra = set(value).difference(expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extra:
                details.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"invalid event fields: {', '.join(details)}")
        return cls(**{key: value[key] for key in expected})

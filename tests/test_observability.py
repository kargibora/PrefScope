from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import stat
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import pytest

from prefscope.core.redaction import (
    DEFAULT_MAX_REDACTION_DEPTH,
    DEFAULT_MAX_REDACTION_NODES,
    DEFAULT_MAX_REDACTION_STRING_LENGTH,
    REDACTED as CORE_REDACTED,
    is_sensitive_key,
    redact_secrets,
    redact_text,
    reject_secrets,
)
from prefscope.observability import (
    DEFAULT_MAX_EVENT_BYTES,
    DEFAULT_MAX_EVENT_NODES,
    DEFAULT_MAX_NESTING_DEPTH,
    DEFAULT_MAX_STRING_LENGTH,
    EVENT_SCHEMA_VERSION,
    REDACTED,
    JsonlRecorder,
    RecorderLoggingHandler,
    RunEvent,
    capture_warnings,
)


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def make_event(**overrides: object) -> RunEvent:
    values = {
        "run_id": "run-1",
        "timestamp": "2026-01-02T03:04:05.000Z",
        "elapsed_seconds": 1.25,
        "stage": "encode",
        "status": "info",
        "message": "working",
        "data": {"items": [1, 2]},
    }
    values.update(overrides)
    return RunEvent(**values)


def read_lines(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_event_is_versioned_immutable_and_round_trips() -> None:
    source = {"items": [1, {"ok": True}]}
    event = make_event(data=source)
    source["items"].append(3)

    assert event.schema_version == EVENT_SCHEMA_VERSION
    assert event.to_dict()["data"] == {"items": [1, {"ok": True}]}
    assert RunEvent.from_dict(event.to_dict()) == event
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.status = "failed"
    with pytest.raises(TypeError):
        event.data["new"] = "value"


def test_run_event_validates_schema_status_time_and_fields() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        make_event(schema_version=99)
    with pytest.raises(ValueError, match="status"):
        make_event(status="success")
    with pytest.raises(ValueError, match="finite and non-negative"):
        make_event(elapsed_seconds=-1)
    with pytest.raises(ValueError, match="end in 'Z'"):
        make_event(timestamp="2026-01-02T03:04:05+00:00")
    serialized = make_event().to_dict()
    serialized["extra"] = True
    with pytest.raises(ValueError, match="unexpected"):
        RunEvent.from_dict(serialized)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_run_event_rejects_non_finite_json_numbers(bad: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        make_event(data={"metric": bad})


def test_run_event_rejects_non_json_values_and_cycles() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        make_event(data={"value": object()})
    with pytest.raises(ValueError, match="keys must be strings"):
        make_event(data={1: "value"})
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="cycles"):
        make_event(data=cyclic)


def test_run_event_rejects_keys_that_collide_after_redaction() -> None:
    with pytest.raises(ValueError, match="collide"):
        make_event(data={"note token=one": 1, "note token=two": 2})


def test_secret_fields_and_common_inline_credentials_are_redacted() -> None:
    event = make_event(
        message='request authorization="abc 123" with Bearer xyz.123',
        data={
            "api_key": "top-secret",
            "nested": {"clientSecret": "also-secret", "token_count": 12},
            "note": "password=hunter2",
        },
    )

    serialized = json.dumps(event.to_dict(), allow_nan=False)
    assert "top-secret" not in serialized
    assert "also-secret" not in serialized
    assert "hunter2" not in serialized
    assert "xyz.123" not in serialized
    assert "abc 123" not in serialized
    assert event.data["api_key"] == REDACTED
    assert event.data["nested"]["token_count"] == 12

    edge_cases = make_event(
        message=(
            "postgresql://alice:supers3cret@db/x "
            "Authorization: Basic dXNlcjpwYXNz "
            "-----BEGIN PRIVATE KEY-----\nABCSECRET"
        ),
        data={"database_url": "postgresql://alice:supers3cret@db/x"},
    )
    edge_serialized = json.dumps(edge_cases.to_dict())
    assert "supers3cret" not in edge_serialized
    assert "dXNlcjpwYXNz" not in edge_serialized
    assert "ABCSECRET" not in edge_serialized


def test_jsonl_recorder_writes_deterministic_valid_event(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    clock = SequenceClock(10.0, 10.5)

    def now() -> datetime:
        return datetime(2026, 1, 2, 3, 4, 5, 678900, tzinfo=timezone.utc)

    with JsonlRecorder(
        path,
        run_id="fixed-run",
        monotonic=clock,
        utc_now=now,
        durable=False,
    ) as recorder:
        returned = recorder.record("load", "started", data={"rows": 4})
        assert path.read_text().endswith("\n")

    rows = read_lines(path)
    assert rows == [returned.to_dict()]
    assert rows[0]["timestamp"] == "2026-01-02T03:04:05.678Z"
    assert rows[0]["elapsed_seconds"] == 0.5
    assert rows[0]["run_id"] == "fixed-run"
    assert rows[0]["status"] == "started"


def test_jsonl_recorder_appends_and_fsyncs_by_default(tmp_path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    fsynced: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: fsynced.append(descriptor))

    with JsonlRecorder(path, run_id="one", monotonic=SequenceClock(0, 1)) as recorder:
        recorder.record("run", "completed")
    with JsonlRecorder(path, run_id="two", monotonic=SequenceClock(2, 3)) as recorder:
        recorder.record("run", "completed")

    assert [row["run_id"] for row in read_lines(path)] == ["one", "two"]
    assert len(fsynced) >= 2


def test_record_failure_redacts_exception_and_preserves_context(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlRecorder(
        path,
        run_id="run",
        durable=False,
        monotonic=SequenceClock(0, 1),
    ) as recorder:
        recorder.record_failure(
            "download",
            RuntimeError("api_key=do-not-store"),
            message="download failed",
            data={"attempt": 2},
        )

    row = read_lines(path)[0]
    assert row["status"] == "failed"
    assert row["message"] == "download failed"
    assert row["data"]["error_type"] == "RuntimeError"
    assert row["data"]["attempt"] == 2
    assert "do-not-store" not in path.read_text()


def test_logging_handler_maps_levels_and_uses_formatter(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    logger = logging.getLogger("prefscope.observability.test")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    with JsonlRecorder(
        path,
        run_id="run",
        durable=False,
        monotonic=SequenceClock(0, 1, 2, 3),
    ) as recorder:
        handler = RecorderLoggingHandler(recorder, stage="worker")
        handler.setFormatter(logging.Formatter("LOG: %(message)s"))
        logger.addHandler(handler)
        logger.info("ready")
        logger.warning("slow")
        logger.error("broken")
        logger.removeHandler(handler)

    rows = read_lines(path)
    assert [row["status"] for row in rows] == ["info", "warning", "failed"]
    assert [row["message"] for row in rows] == [
        "LOG: ready",
        "LOG: slow",
        "LOG: broken",
    ]
    assert all(row["data"]["logger"] == logger.name for row in rows)


def test_warning_bridge_records_and_restores_standard_hook(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    original = warnings.showwarning
    with JsonlRecorder(
        path,
        run_id="run",
        durable=False,
        monotonic=SequenceClock(0, 1),
    ) as recorder:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with capture_warnings(recorder, stage="validation", forward=False):
                warnings.warn("deprecated option", DeprecationWarning)
            assert warnings.showwarning is original

    row = read_lines(path)[0]
    assert row["stage"] == "validation"
    assert row["status"] == "warning"
    assert row["message"] == "deprecated option"
    assert row["data"]["category"] == "DeprecationWarning"
    assert row["data"]["filename"] == "test_observability.py"
    assert "/" not in row["data"]["filename"]
    assert isinstance(row["data"]["lineno"], int)


def test_warning_bridge_forwards_by_default_and_is_best_effort(
    tmp_path, monkeypatch
) -> None:
    recorder = JsonlRecorder(tmp_path / "events.jsonl", durable=False)
    recorder.close()
    forwarded: list[str] = []

    def prior_showwarning(message, category, filename, lineno, file=None, line=None):
        forwarded.append(str(message))

    monkeypatch.setattr(warnings, "showwarning", prior_showwarning)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with capture_warnings(recorder):
            warnings.warn("still visible", UserWarning)

    assert forwarded == ["still visible"]


def test_closed_recorder_refuses_more_events(tmp_path) -> None:
    recorder = JsonlRecorder(tmp_path / "events.jsonl", durable=False)
    recorder.close()
    with pytest.raises(ValueError, match="closed"):
        recorder.record("run", "info")


def test_observability_import_does_not_load_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import prefscope.observability; "
                "assert not any(name == 'torch' or name.startswith('torch.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "key",
    [
        "accessToken",
        "AccountKey",
        "account_key",
        "connectionString",
        "connectionStrings",
        "conn_str",
        "defaultConnection",
        "hfToken",
        "authToken",
        "authorization_header",
        "aws_access_key_id",
        "awsAccessKeyId",
        "secretAccessKey",
        "sharedAccessSignature",
        "clientSecret",
    ],
)
def test_central_redaction_normalizes_sensitive_key_variants(key: str) -> None:
    assert is_sensitive_key(key)
    assert redact_secrets({key: "must-not-remain"})[key] == CORE_REDACTED
    with pytest.raises(ValueError, match="credential-like field") as caught:
        reject_secrets({key: "must-not-remain"}, where="provenance")
    assert "must-not-remain" not in str(caught.value)


def test_central_redaction_keeps_harmless_token_metrics() -> None:
    value = {"token_count": 12, "promptTokens": 5, "max_tokens": 100}
    assert redact_secrets(value) == value
    assert reject_secrets(value, where="metrics") is value


@pytest.mark.parametrize(
    "secret_text",
    [
        "Bearer abcdefghijklmnop",
        "Basic dXNlcjpwYXNz",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.signature123",
        "hf_abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz",
        "github_pat_abcdefghijklmnopqrstuvwxyz",
        "glpat-abcdefghijklmnop",
        "AKIAABCDEFGHIJKLMNOP",
        "AIzaSyD-abcdefghijklmnop",
        "xoxb-12345678-abcdefgh",
        "sk-proj-abcdefghijklmnop",
        "rk_live_abcdefghijklmnop",
        "postgresql://alice:supers3cret@db.example/x",
        "accessToken=abcdefghijklmnop",
        "AccountKey=abcdefghijklmnop",
        "Server=db;AccountKey=abcdefghijklmnop",
        "https://blob.example/x?sig=abcdefghijklmnop",
        "https://blob.example/x?key=abcdefghijklmnop",
        "https://blob.example/x?token=abcdefghijklmnop",
        "person@example.org",
        "person@[192.0.2.1]",
        "+1 (202) 555-0123",
        "+12025550123",
        "-----BEGIN PRIVATE KEY-----\nABCSECRET",
    ],
)
def test_central_text_scanner_redacts_and_rejects_bypasses(secret_text: str) -> None:
    redacted = redact_text(f"prefix {secret_text}")
    assert secret_text not in redacted
    assert CORE_REDACTED in redacted
    with pytest.raises(ValueError, match="credential-like text") as caught:
        reject_secrets({"note": secret_text}, where="report")
    assert secret_text not in str(caught.value)


def test_event_redacts_recursive_compact_keys_and_text_credentials() -> None:
    event = make_event(
        data={
            "nested": [{"accessToken": "one", "token_count": 2}],
            "note": "Basic dXNlcjpwYXNz",
        }
    )
    serialized = json.dumps(event.to_dict())
    assert "one" not in serialized
    assert "dXNlcjpwYXNz" not in serialized
    assert event.data["nested"][0]["token_count"] == 2


def test_central_policy_redacts_pii_and_connection_secrets_across_events() -> None:
    raw = {
        "connectionString": "Server=db;AccountKey=do-not-store",
        "contact": "person@[192.0.2.1] or +1 (202) 555-0123",
        "email": "person@example.org",
    }
    redacted = redact_secrets(raw)
    event = make_event(data=raw)
    for serialized in (json.dumps(redacted), json.dumps(event.to_dict())):
        assert "do-not-store" not in serialized
        assert "person@" not in serialized
        assert "555-0123" not in serialized
    with pytest.raises(ValueError, match="credential"):
        reject_secrets(raw, where="provenance")


def test_event_enforces_string_nesting_and_node_limits() -> None:
    with pytest.raises(ValueError, match="maximum string length"):
        make_event(message="x" * (DEFAULT_MAX_STRING_LENGTH + 1))

    nested: object = 0
    for _ in range(DEFAULT_MAX_NESTING_DEPTH + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="maximum nesting depth"):
        make_event(data={"nested": nested})

    with pytest.raises(ValueError, match="maximum node count"):
        make_event(data={"items": [0] * DEFAULT_MAX_EVENT_NODES})


def test_recorder_rejects_oversized_serialized_event_without_writing(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlRecorder(
        path,
        durable=False,
        max_event_bytes=100,
        monotonic=SequenceClock(0, 1),
    ) as recorder:
        with pytest.raises(ValueError, match="maximum size"):
            recorder.record("stage", "info", "x")
    assert path.read_bytes() == b""
    assert DEFAULT_MAX_EVENT_BYTES > 100


def test_recorder_creates_parent_and_forces_private_permissions(tmp_path) -> None:
    path = tmp_path / "nested" / "events.jsonl"
    with JsonlRecorder(path, durable=False):
        pass
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    path.chmod(0o644)
    with JsonlRecorder(path, durable=False):
        pass
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_recorder_rejects_symlink_without_touching_target(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("unchanged")
    link = tmp_path / "trace.jsonl"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        JsonlRecorder(link, durable=False)
    assert target.read_text() == "unchanged"


def test_recorder_rejects_symlink_parent_without_creating_trace(tmp_path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe component"):
        JsonlRecorder(linked_parent / "events.jsonl", durable=False)
    assert not (real_parent / "events.jsonl").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is not available")
def test_recorder_rejects_fifo_without_blocking(tmp_path) -> None:
    fifo = tmp_path / "trace.pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        JsonlRecorder(fifo, durable=False)


def test_recorder_descriptor_is_close_on_exec(tmp_path) -> None:
    with JsonlRecorder(tmp_path / "events.jsonl", durable=False) as recorder:
        assert not os.get_inheritable(recorder._descriptor)


def test_recorder_fsyncs_parent_when_creating_file(tmp_path, monkeypatch) -> None:
    fsynced: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: fsynced.append(descriptor))
    with JsonlRecorder(tmp_path / "new" / "events.jsonl", durable=False):
        pass
    assert len(fsynced) >= 1


@pytest.mark.parametrize("operation", ["redact", "reject"])
def test_recursive_secret_helpers_enforce_depth_node_and_string_limits(
    operation,
) -> None:
    def apply(value):
        if operation == "redact":
            return redact_secrets(value)
        return reject_secrets(value, where="report")

    nested: object = 0
    for _ in range(DEFAULT_MAX_REDACTION_DEPTH + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="maximum secret-scan depth"):
        apply(nested)

    with pytest.raises(ValueError, match="maximum secret-scan node count"):
        apply([0] * DEFAULT_MAX_REDACTION_NODES)
    if operation == "redact":
        with pytest.raises(ValueError, match="maximum secret-scan node count"):
            apply({"apiKey": [0] * DEFAULT_MAX_REDACTION_NODES})

    oversized = "x" * (DEFAULT_MAX_REDACTION_STRING_LENGTH + 1)
    with pytest.raises(ValueError, match="maximum secret-scan string length"):
        apply({"note": oversized})
    with pytest.raises(ValueError, match="maximum secret-scan key length"):
        apply({oversized: "value"})


@pytest.mark.parametrize("swap_phase", ["before_open", "after_open"])
def test_recorder_rejects_existing_file_identity_races(
    tmp_path, monkeypatch, swap_phase
) -> None:
    trace = tmp_path / "events.jsonl"
    trace.write_text("old")
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text("new")
    original_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        is_trace_open = (
            not swapped
            and path == trace.name
            and bool(flags & os.O_WRONLY)
            and dir_fd is not None
        )
        if is_trace_open and swap_phase == "before_open":
            os.replace(replacement, trace)
            swapped = True
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if is_trace_open and swap_phase == "after_open":
            os.replace(replacement, trace)
            swapped = True
        return descriptor

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(ValueError, match="changed while opening"):
        JsonlRecorder(trace, durable=False)

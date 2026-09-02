from __future__ import annotations

import builtins
import io
import json
import os
import subprocess
import sys
import threading
import warnings
import pytest

from prefscope.observability import observe_run
from prefscope.observability.console import ConsoleReporter
from prefscope.observability.events import RunEvent
from prefscope.observability import runtime as runtime_module
from prefscope.observability.runtime import automatic_stage


def _event(
    *,
    stage: str = "load_lens",
    status: str = "completed",
    message: str = "",
    data=None,
) -> RunEvent:
    return RunEvent(
        run_id="test-run",
        timestamp="2026-01-02T03:04:05Z",
        elapsed_seconds=1.0,
        stage=stage,
        status=status,
        message=message,
        data={} if data is None else data,
    )


def _rows(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    runtime_module._close_environment_recorder()
    monkeypatch.delenv("PREFSCOPE_EVENTS_PATH", raising=False)
    monkeypatch.delenv("PREFSCOPE_EVENTS_PRETTY", raising=False)
    yield
    runtime_module._close_environment_recorder()


def test_explicit_pretty_prints_one_compact_terminal_line(tmp_path, capsys) -> None:
    path = tmp_path / "events.jsonl"
    with observe_run(path, pretty=True, durable=False):
        with automatic_stage("load_lens") as span:
            span.update(n_rows=3, n_features=24_576)

    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("✓ Load lens  ")
    assert "3 rows" in lines[0]
    assert "24,576 features" in lines[0]
    assert [row["status"] for row in _rows(path)] == ["started", "completed"]


def test_environment_pretty_truthy_and_invalid_values(
    tmp_path, monkeypatch, capsys
) -> None:
    path = tmp_path / "truthy.jsonl"
    monkeypatch.setenv("PREFSCOPE_EVENTS_PATH", str(path))
    monkeypatch.setenv("PREFSCOPE_EVENTS_PRETTY", "YeS")
    with automatic_stage("encode"):
        pass
    runtime_module._close_environment_recorder()
    assert "✓ Encode" in capsys.readouterr().err

    invalid_path = tmp_path / "invalid.jsonl"
    monkeypatch.setenv("PREFSCOPE_EVENTS_PATH", str(invalid_path))
    monkeypatch.setenv("PREFSCOPE_EVENTS_PRETTY", "maybe")
    with automatic_stage("encode"):
        pass
    runtime_module._close_environment_recorder()
    assert capsys.readouterr().err == ""
    assert len(_rows(invalid_path)) == 2


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "invalid"])
def test_environment_pretty_false_values_are_silent(
    value, tmp_path, monkeypatch, capsys
) -> None:
    path = tmp_path / f"false-{value or 'empty'}.jsonl"
    monkeypatch.setenv("PREFSCOPE_EVENTS_PATH", str(path))
    monkeypatch.setenv("PREFSCOPE_EVENTS_PRETTY", value)
    with automatic_stage("encode"):
        pass
    runtime_module._close_environment_recorder()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert len(_rows(path)) == 2


def test_pretty_environment_without_event_path_is_a_noop(monkeypatch, capsys) -> None:
    monkeypatch.setenv("PREFSCOPE_EVENTS_PRETTY", "1")
    with automatic_stage("encode") as span:
        assert not span.active
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert runtime_module._ENVIRONMENT_RUN is None


def test_observe_run_default_consults_env_and_false_overrides(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("PREFSCOPE_EVENTS_PRETTY", "on")
    with observe_run(tmp_path / "default.jsonl", durable=False):
        with automatic_stage("load_lens"):
            pass
    assert "✓ Load lens" in capsys.readouterr().err

    with observe_run(tmp_path / "override.jsonl", pretty=False, durable=False):
        with automatic_stage("load_lens"):
            pass
    assert capsys.readouterr().err == ""


def test_pretty_false_has_no_output_and_validates_before_file_creation(
    tmp_path, capsys
) -> None:
    path = tmp_path / "disabled.jsonl"
    with observe_run(path, pretty=False, durable=False):
        with automatic_stage("load_lens"):
            pass
    assert capsys.readouterr().err == ""

    invalid_path = tmp_path / "invalid-pretty.jsonl"
    with pytest.raises(ValueError, match="pretty must be a bool or None"):
        with observe_run(invalid_path, pretty=1):
            pass
    assert not invalid_path.exists()


def test_completed_failed_and_warning_output_is_privacy_safe(tmp_path, capsys) -> None:
    path = tmp_path / "privacy.jsonl"
    raw = "PRIVATE prompt /Users/alice/secret.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with observe_run(path, pretty=True, durable=False):
            with automatic_stage("encode") as span:
                span.update(n_rows=2)
            with pytest.raises(RuntimeError, match="PRIVATE"):
                with automatic_stage("load_lens"):
                    raise RuntimeError(raw)
            warnings.warn(raw, UserWarning)

    output = capsys.readouterr().err
    assert "✓ Encode" in output
    assert "✗ Load lens" in output
    assert "RuntimeError" in output
    assert "⚠ UserWarning" in output
    assert raw not in output
    assert "/Users/alice" not in output


def test_untrusted_manual_and_automatic_fields_are_never_rendered(
    tmp_path, capsys
) -> None:
    path = tmp_path / "untrusted.jsonl"
    raw = "CUSTOM_PRIVATE_STAGE /Users/private/name"

    with observe_run(path, pretty=True, durable=False) as run:
        run.record(
            raw,
            "completed",
            raw,
            {"name": raw, "profile": raw, "status": raw, "n_rows": 2},
        )
        run.record(raw, "warning", raw, {"category": raw})
        with automatic_stage(raw, {"name": raw}):
            pass

    output = capsys.readouterr().err
    assert output.splitlines()[0] == "✓ Operation  2 rows"
    assert output.splitlines()[1] == "⚠ Warning"
    assert output.splitlines()[2].startswith("✓ Operation  ")
    assert raw not in output
    assert "/Users/private" not in output
    assert len(_rows(path)) == 4


def test_console_output_is_bounded_with_one_suppression_notice(
    tmp_path, capsys
) -> None:
    path = tmp_path / "bounded.jsonl"
    with observe_run(path, pretty=True, durable=False) as run:
        for _ in range(205):
            run.record("load_lens", "completed")

    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 201
    assert lines[:200] == ["✓ Load lens"] * 200
    assert lines[200] == "… Further observability output suppressed"
    assert len(_rows(path)) == 205


def test_rich_terminal_uses_code_owned_styles_without_leaking_input(
    monkeypatch,
) -> None:
    pytest.importorskip("rich.console")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, max_lines=3, _force_terminal=True)
    raw = "[bold red]PRIVATE /Users/alice/secret.txt[/bold red]"

    reporter.observe(_event(stage=raw, message=raw, data={"n_rows": 1, "path": raw}))
    reporter.observe(
        _event(
            stage=raw,
            status="failed",
            message=raw,
            data={"error_type": raw},
        )
    )
    reporter.observe(_event(status="warning", message=raw, data={"category": raw}))
    reporter.observe(_event())

    output = stream.getvalue()
    assert "\x1b[32m✓ Operation  1 row\x1b[0m" in output
    assert "\x1b[31m✗ Operation  Exception\x1b[0m" in output
    assert "\x1b[33m⚠ Warning\x1b[0m" in output
    assert "\x1b[2m… Further observability output suppressed\x1b[0m" in output
    assert raw not in output
    assert "/Users/alice" not in output


def test_rich_import_failure_uses_plain_stderr(monkeypatch) -> None:
    stream = io.StringIO()
    original_import = builtins.__import__

    def missing_rich(name, *args, **kwargs):
        if name == "rich.console":
            raise ImportError("Rich unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_rich)
    reporter = ConsoleReporter(stream=stream)
    reporter.observe(_event(data={"duration_seconds": 4.23, "n_rows": 3}))
    assert stream.getvalue() == "✓ Load lens  4.23s · 3 rows\n"


def test_hostile_huge_numeric_details_are_omitted() -> None:
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream)
    reporter.observe(_event(data={"n_rows": 10**5_000, "duration_seconds": 1e308}))
    assert stream.getvalue() == "✓ Load lens\n"


def test_reporter_base_exceptions_do_not_change_recording_or_application(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []

    class BrokenReporter:
        def observe(self, event) -> None:
            calls.append(event.status)
            raise KeyboardInterrupt("render failure")

        def close(self) -> None:
            calls.append("closed")
            raise KeyboardInterrupt("close failure")

    monkeypatch.setattr(
        runtime_module, "_create_console_reporter", lambda: BrokenReporter()
    )
    path = tmp_path / "reporter-failure.jsonl"
    reached = False
    with observe_run(path, pretty=True, durable=False):
        with automatic_stage("encode"):
            reached = True

    assert reached
    assert calls == ["started", "completed", "closed"]
    assert [row["status"] for row in _rows(path)] == ["started", "completed"]


def test_manual_record_failure_dispatch_is_privacy_safe(tmp_path, capsys) -> None:
    class PrivateFailure(Exception):
        pass

    raw = "PRIVATE_FAILURE /Users/alice/private.txt"
    path = tmp_path / "manual-failure.jsonl"
    with observe_run(path, pretty=True, durable=False) as run:
        run.record_failure(
            raw,
            PrivateFailure(raw),
            message=raw,
            data={"n_rows": 2, "path": raw},
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "✗ Operation  2 rows · Exception"
    assert raw not in captured.err


def test_console_lines_are_atomic_across_threads() -> None:
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, max_lines=200)
    event = _event()

    def emit() -> None:
        for _ in range(25):
            reporter.observe(event)

    threads = [threading.Thread(target=emit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert stream.getvalue().splitlines() == ["✓ Load lens"] * 200


def test_disabled_path_does_not_load_presenter_or_torch_in_subprocess(tmp_path) -> None:
    path = tmp_path / "subprocess.jsonl"
    code = f"""
import sys
from prefscope.observability import observe_run
from prefscope.observability.runtime import automatic_stage
rich_before = {{name for name in sys.modules if name.startswith('rich')}}
with observe_run({os.fspath(path)!r}, pretty=False, durable=False):
    with automatic_stage('encode'):
        pass
rich_after = {{name for name in sys.modules if name.startswith('rich')}}
assert 'prefscope.observability.console' not in sys.modules
assert rich_after == rich_before
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.fspath(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone

import pytest

from prefscope.observability import RunContext, observe_run
from prefscope.observability import runtime as runtime_module
from prefscope.observability.runtime import automatic_stage


class StepClock:
    def __init__(self, *, step: float = 0.25) -> None:
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def fixed_now() -> datetime:
    return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def read_events(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def clean_environment_runtime(monkeypatch):
    runtime_module._close_environment_recorder()
    monkeypatch.delenv("PREFSCOPE_EVENTS_PATH", raising=False)
    yield
    runtime_module._close_environment_recorder()


def test_automatic_stage_records_success_and_explicit_result_data(
    tmp_path, monkeypatch
) -> None:
    ids = iter(["operation-1"])
    monkeypatch.setattr(runtime_module.uuid, "uuid4", lambda: next(ids))
    path = tmp_path / "events.jsonl"

    with observe_run(
        path,
        run_id="run-1",
        durable=False,
        monotonic=StepClock(),
        utc_now=fixed_now,
    ) as run:
        assert isinstance(run, RunContext)
        with automatic_stage("encode", {"input_rows": 4}) as operation:
            assert operation.active
            operation.update(output_rows=3, shape=[3, 8])

    rows = read_events(path)
    assert [row["status"] for row in rows] == ["started", "completed"]
    assert {row["run_id"] for row in rows} == {"run-1"}
    assert rows[0]["data"] == {
        "input_rows": 4,
        "operation_id": "operation-1",
        "parent_operation_id": None,
    }
    assert rows[1]["data"]["output_rows"] == 3
    assert rows[1]["data"]["shape"] == [3, 8]
    assert rows[1]["data"]["duration_seconds"] == pytest.approx(0.5)
    assert rows[1]["data"]["operation_id"] == "operation-1"
    assert run.closed


def test_automatic_stage_records_sanitized_failure_and_reraises(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_module.uuid, "uuid4", lambda: "failed-operation")
    path = tmp_path / "events.jsonl"

    with pytest.raises(RuntimeError, match="do-not-store"):
        with observe_run(
            path,
            durable=False,
            monotonic=StepClock(),
            utc_now=fixed_now,
        ):
            with automatic_stage("download", {"attempt": 2}):
                raise RuntimeError("api_key=do-not-store")

    rows = read_events(path)
    assert [row["status"] for row in rows] == ["started", "failed"]
    failure = rows[1]
    assert failure["message"] == "download failed"
    assert failure["data"]["error_type"] == "RuntimeError"
    assert "error_message" not in failure["data"]
    assert failure["data"]["operation_id"] == "failed-operation"
    assert failure["data"]["parent_operation_id"] is None
    assert failure["data"]["duration_seconds"] == pytest.approx(0.5)
    assert "do-not-store" not in path.read_text()
    assert "Traceback" not in path.read_text()


def test_nested_operations_and_explicit_runs_restore_parent_context(
    tmp_path, monkeypatch
) -> None:
    ids = iter(["outer-operation", "inner-run-operation", "outer-child"])
    monkeypatch.setattr(runtime_module.uuid, "uuid4", lambda: next(ids))
    outer_path = tmp_path / "outer.jsonl"
    inner_path = tmp_path / "inner.jsonl"

    with observe_run(outer_path, run_id="outer-run", durable=False):
        with automatic_stage("outer"):
            with observe_run(inner_path, run_id="inner-run-id", durable=False):
                with automatic_stage("inner-run"):
                    pass
            with automatic_stage("outer-child"):
                pass

    outer = read_events(outer_path)
    inner = read_events(inner_path)
    outer_started = {row["stage"]: row for row in outer if row["status"] == "started"}
    assert outer_started["outer"]["data"]["parent_operation_id"] is None
    assert (
        outer_started["outer-child"]["data"]["parent_operation_id"] == "outer-operation"
    )
    assert inner[0]["data"]["parent_operation_id"] is None
    assert {row["stage"] for row in outer} == {"outer", "outer-child"}
    assert {row["stage"] for row in inner} == {"inner-run"}


def test_environment_activation_is_lazy_and_no_configuration_is_a_noop(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"

    with automatic_stage("disabled") as operation:
        assert not operation.active
        operation.update(arbitrary_object=object())
        with pytest.raises(AttributeError):
            operation.active = True
    assert not path.exists()

    monkeypatch.setenv("PREFSCOPE_EVENTS_PATH", str(path))
    with automatic_stage("enabled", {"rows": 2}) as operation:
        assert operation.active
    environment_run = runtime_module._ENVIRONMENT_RUN
    assert environment_run is not None
    assert not environment_run.closed

    runtime_module._close_environment_recorder()
    assert environment_run.closed
    assert runtime_module._ENVIRONMENT_RUN is None
    assert [row["status"] for row in read_events(path)] == ["started", "completed"]


def test_explicit_context_bridges_prefscope_logs_and_warnings_then_cleans_up(
    tmp_path,
) -> None:
    path = tmp_path / "events.jsonl"
    logger = logging.getLogger("prefscope.runtime-test")
    original_level = logger.level
    original_propagate = logger.propagate
    original_handlers = list(logger.handlers)
    original_showwarning = warnings.showwarning
    root_handlers = list(logging.getLogger("prefscope").handlers)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.INFO)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with observe_run(path, durable=False):
                logger.info("processed 4 rows")
                warnings.warn("fallback used", UserWarning)
        size_after_close = path.stat().st_size
        logger.warning("outside context")
    finally:
        logger.handlers[:] = original_handlers
        logger.propagate = original_propagate
        logger.setLevel(original_level)

    rows = read_events(path)
    assert [(row["stage"], row["status"]) for row in rows] == [
        ("logging", "info"),
        ("warnings", "warning"),
    ]
    assert rows[0]["data"]["logger"] == "prefscope"
    assert rows[1]["data"]["category"] == "UserWarning"
    assert rows[0]["message"] == rows[1]["message"] == ""
    assert set(rows[1]["data"]) == {
        "category",
        "operation_id",
        "parent_operation_id",
    }
    assert path.stat().st_size == size_after_close
    assert warnings.showwarning is original_showwarning
    assert logging.getLogger("prefscope").handlers == root_handlers


def test_result_data_uses_event_bounds_and_redaction(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    opaque_argument = object()

    with observe_run(path, durable=False):
        with automatic_stage("safe", {"token": "top-secret"}) as operation:
            # Instrumentation records only data selected by the call site. It
            # never receives or introspects opaque_argument.
            assert opaque_argument is not None
            operation.set_result_data(
                {"authorization": "Bearer do-not-store", "output_rows": 1}
            )

    text = path.read_text()
    rows = read_events(path)
    assert "top-secret" not in text
    assert "do-not-store" not in text
    assert "opaque_argument" not in text
    assert rows[1]["data"]["output_rows"] == 1


def test_invalid_automatic_event_data_does_not_mask_operation(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    completed = False
    with observe_run(path, durable=False):
        with automatic_stage("unsafe", {"raw": object()}):
            completed = True

    assert completed
    assert path.read_bytes() == b""


def test_automatic_bridge_omits_raw_log_warning_and_exception_text(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    raw = "RAW_PROMPT /Users/alice/private/dataset.csv row secret"
    forwarded: list[str] = []

    def prior_showwarning(message, category, filename, lineno, file=None, line=None):
        forwarded.append(str(message))

    monkeypatch.setattr(warnings, "showwarning", prior_showwarning)
    logger = logging.getLogger("prefscope.adversarial")
    original_level = logger.level
    original_propagate = logger.propagate
    original_handlers = list(logger.handlers)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.INFO)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with observe_run(path, durable=False):
                try:
                    raise RuntimeError(raw)
                except RuntimeError:
                    logger.exception("failed with %s", raw)
                warnings.warn(raw, UserWarning)
                with pytest.raises(ValueError, match="RAW_PROMPT"):
                    with automatic_stage("encode"):
                        raise ValueError(raw)
    finally:
        logger.handlers[:] = original_handlers
        logger.propagate = original_propagate
        logger.setLevel(original_level)

    text = path.read_text()
    assert raw not in text
    assert "/Users/alice" not in text
    rows = read_events(path)
    logging_row = next(row for row in rows if row["stage"] == "logging")
    warning_row = next(row for row in rows if row["stage"] == "warnings")
    failure_row = next(
        row for row in rows if row["status"] == "failed" and row["stage"] == "encode"
    )
    assert logging_row["message"] == ""
    assert set(logging_row["data"]) == {
        "logger",
        "level",
        "error_type",
        "operation_id",
        "parent_operation_id",
    }
    assert warning_row["message"] == ""
    assert set(warning_row["data"]) == {
        "category",
        "operation_id",
        "parent_operation_id",
    }
    assert failure_row["message"] == "encode failed"
    assert "error_message" not in failure_row["data"]
    assert forwarded == [raw]


def test_automatic_recorder_errors_do_not_change_success_or_failure(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    original = RuntimeError("original application failure")

    def fail_record(*args, **kwargs):
        raise KeyboardInterrupt("recorder unavailable")

    with observe_run(path, durable=False) as run:
        monkeypatch.setattr(run, "record", fail_record)
        with automatic_stage("successful"):
            pass
        with pytest.raises(RuntimeError) as caught:
            with automatic_stage("failed"):
                raise original

    assert caught.value is original
    assert path.read_bytes() == b""


def test_custom_observation_identifiers_are_normalized(tmp_path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    raw = "PRIVATE_TENANT prompt /Users/private/customer.csv"
    forwarded: list[str] = []

    class PrivateTenantFailure(Exception):
        pass

    class PrivateTenantWarning(Warning):
        pass

    def prior_showwarning(message, category, filename, lineno, file=None, line=None):
        forwarded.append(str(message))

    monkeypatch.setattr(warnings, "showwarning", prior_showwarning)
    prefscope_logger = logging.getLogger("prefscope")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with observe_run(path, durable=False):
            try:
                raise PrivateTenantFailure(raw)
            except PrivateTenantFailure:
                record = prefscope_logger.makeRecord(
                    "prefscope.PRIVATE_TENANT_logger",
                    logging.WARNING + 5,
                    raw,
                    1,
                    raw,
                    (),
                    sys.exc_info(),
                )
                record.levelname = "PRIVATE_TENANT_LEVEL"
                prefscope_logger.handle(record)
            warnings.warn(raw, PrivateTenantWarning)
            with pytest.raises(PrivateTenantFailure):
                with automatic_stage("custom-failure"):
                    raise PrivateTenantFailure(raw)

    text = path.read_text()
    assert "PRIVATE_TENANT" not in text
    assert "/Users/private" not in text
    rows = read_events(path)
    logging_row = next(row for row in rows if row["stage"] == "logging")
    warning_row = next(row for row in rows if row["stage"] == "warnings")
    failure_row = next(
        row
        for row in rows
        if row["stage"] == "custom-failure" and row["status"] == "failed"
    )
    assert logging_row["data"]["logger"] == "prefscope"
    assert logging_row["data"]["level"] == "WARNING"
    assert logging_row["data"]["error_type"] == "Exception"
    assert warning_row["data"]["category"] == "Warning"
    assert failure_row["data"]["error_type"] == "Exception"
    assert forwarded == [raw]


def test_explicit_close_errors_do_not_change_success_or_failure(
    tmp_path, monkeypatch
) -> None:
    success_path = tmp_path / "success.jsonl"
    success_reached = False
    with observe_run(success_path, durable=False) as run:
        original_close = run.close

        def close_then_fail() -> None:
            original_close()
            raise OSError("close failed")

        monkeypatch.setattr(run, "close", close_then_fail)
        success_reached = True
    assert success_reached

    failure_path = tmp_path / "failure.jsonl"
    original_error = RuntimeError("application failed")
    with pytest.raises(RuntimeError) as caught:
        with observe_run(failure_path, durable=False) as run:
            original_close = run.close

            def close_then_fail_again() -> None:
                original_close()
                raise OSError("close failed")

            monkeypatch.setattr(run, "close", close_then_fail_again)
            raise original_error
    assert caught.value is original_error


def test_operation_id_failure_yields_inactive_span_and_executes_body(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"

    def fail_uuid():
        raise RuntimeError("UUID unavailable")

    monkeypatch.setattr(runtime_module.uuid, "uuid4", fail_uuid)
    body_reached = False
    with observe_run(path, run_id="fixed-run", durable=False):
        with automatic_stage("encode") as operation:
            assert not operation.active
            body_reached = True
    assert body_reached
    assert path.read_bytes() == b""


def test_bridge_cleanup_errors_do_not_change_success_or_failure(
    tmp_path, monkeypatch
) -> None:
    original_remove = runtime_module._remove_bridges

    def remove_then_fail() -> None:
        original_remove()
        raise OSError("bridge cleanup failed")

    monkeypatch.setattr(runtime_module, "_remove_bridges", remove_then_fail)
    success_reached = False
    with observe_run(tmp_path / "success-bridge.jsonl", durable=False):
        success_reached = True
    assert success_reached

    original_error = RuntimeError("application failed")
    with pytest.raises(RuntimeError) as caught:
        with observe_run(tmp_path / "failure-bridge.jsonl", durable=False):
            raise original_error
    assert caught.value is original_error


def test_bridge_install_failure_restores_context_and_closes_recorder(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "install-failure.jsonl"
    closed: list[bool] = []
    original_close = runtime_module.JsonlRecorder.close

    def close_spy(recorder) -> None:
        original_close(recorder)
        closed.append(True)

    def fail_install() -> None:
        raise RuntimeError("bridge install failed")

    monkeypatch.setattr(runtime_module.JsonlRecorder, "close", close_spy)
    monkeypatch.setattr(runtime_module, "_install_bridges", fail_install)
    with pytest.raises(RuntimeError, match="bridge install failed"):
        with observe_run(path, durable=False):
            pytest.fail("context body must not run")

    assert closed == [True]
    assert runtime_module._CURRENT_RUN.get() is None
    with automatic_stage("after-failed-entry") as operation:
        assert not operation.active


def test_environment_bridge_install_failure_is_a_noop(tmp_path, monkeypatch) -> None:
    path = tmp_path / "environment-install-failure.jsonl"
    monkeypatch.setenv("PREFSCOPE_EVENTS_PATH", str(path))

    def fail_install() -> None:
        raise RuntimeError("bridge install failed")

    monkeypatch.setattr(runtime_module, "_install_bridges", fail_install)
    body_reached = False
    with automatic_stage("environment") as operation:
        assert not operation.active
        body_reached = True
    assert body_reached
    assert runtime_module._ENVIRONMENT_RUN is None
    assert path.read_bytes() == b""


def test_partial_bridge_install_is_transactional(tmp_path, monkeypatch) -> None:
    path = tmp_path / "partial-install.jsonl"
    logger = logging.getLogger("prefscope")
    original_handlers = list(logger.handlers)
    original_showwarning = warnings.showwarning
    original_add = logger.addHandler

    def add_then_fail(handler) -> None:
        original_add(handler)
        raise RuntimeError("partial install")

    monkeypatch.setattr(logger, "addHandler", add_then_fail)
    with pytest.raises(RuntimeError, match="partial install"):
        with observe_run(path, durable=False):
            pytest.fail("context body must not run")
    assert logger.handlers == original_handlers
    assert warnings.showwarning is original_showwarning
    assert runtime_module._BRIDGE_USERS == 0
    assert runtime_module._CURRENT_RUN.get() is None


def test_partial_bridge_remove_still_restores_all_state(tmp_path, monkeypatch) -> None:
    path = tmp_path / "partial-remove.jsonl"
    logger = logging.getLogger("prefscope")
    original_handlers = list(logger.handlers)
    original_showwarning = warnings.showwarning

    def fail_remove(handler) -> None:
        raise RuntimeError("remove failed")

    with observe_run(path, durable=False):
        monkeypatch.setattr(logger, "removeHandler", fail_remove)
    assert logger.handlers == original_handlers
    assert warnings.showwarning is original_showwarning
    assert runtime_module._BRIDGE_USERS == 0


def test_operation_stack_set_failure_yields_inactive_and_runs(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "stack-set.jsonl"
    original_stack = runtime_module._OPERATION_STACK

    class FailingSetStack:
        def get(self):
            return original_stack.get()

        def set(self, value):
            raise KeyboardInterrupt("stack set failed")

    body_reached = False
    with observe_run(path, durable=False):
        with monkeypatch.context() as context_patch:
            context_patch.setattr(runtime_module, "_OPERATION_STACK", FailingSetStack())
            with automatic_stage("stack-set") as operation:
                assert not operation.active
                body_reached = True
    assert body_reached
    assert path.read_bytes() == b""


def test_operation_stack_reset_failure_preserves_success_and_error(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "stack-reset.jsonl"
    original_stack = runtime_module._OPERATION_STACK

    class FailingResetStack:
        def get(self):
            return original_stack.get()

        def set(self, value):
            return original_stack.set(value)

        def reset(self, token):
            raise KeyboardInterrupt("stack reset failed")

    success_reached = False
    with observe_run(path, durable=False):
        with monkeypatch.context() as context_patch:
            context_patch.setattr(
                runtime_module, "_OPERATION_STACK", FailingResetStack()
            )
            with automatic_stage("success-reset"):
                success_reached = True
    assert success_reached

    original_error = RuntimeError("application failure")
    with pytest.raises(RuntimeError) as caught:
        with observe_run(tmp_path / "failure-reset.jsonl", durable=False):
            with monkeypatch.context() as context_patch:
                context_patch.setattr(
                    runtime_module, "_OPERATION_STACK", FailingResetStack()
                )
                with automatic_stage("failure-reset"):
                    raise original_error
    assert caught.value is original_error


def test_closed_explicit_run_yields_inactive_span(tmp_path) -> None:
    path = tmp_path / "closed-run.jsonl"
    with observe_run(path, durable=False) as run:
        run.close()
        with automatic_stage("closed") as operation:
            assert not operation.active
    assert path.read_bytes() == b""


def test_environment_recorder_base_exception_is_a_noop(tmp_path, monkeypatch) -> None:
    path = tmp_path / "environment-base-exception.jsonl"
    monkeypatch.setenv("PREFSCOPE_EVENTS_PATH", str(path))

    def fail_recorder(*args, **kwargs):
        raise KeyboardInterrupt("recorder construction failed")

    monkeypatch.setattr(runtime_module, "JsonlRecorder", fail_recorder)
    body_reached = False
    with automatic_stage("environment-construction") as operation:
        assert not operation.active
        body_reached = True
    assert body_reached
    assert not path.exists()


def test_observe_run_second_context_set_failure_rolls_back_everything(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "second-set-failure.jsonl"
    original_stack = runtime_module._OPERATION_STACK
    logger = logging.getLogger("prefscope")
    original_handlers = list(logger.handlers)
    original_showwarning = warnings.showwarning
    closed: list[bool] = []
    original_close = runtime_module.JsonlRecorder.close

    class FailingStack:
        def set(self, value):
            raise RuntimeError("second ContextVar set failed")

    def close_spy(recorder) -> None:
        original_close(recorder)
        closed.append(True)

    monkeypatch.setattr(runtime_module.JsonlRecorder, "close", close_spy)
    with monkeypatch.context() as context_patch:
        context_patch.setattr(runtime_module, "_OPERATION_STACK", FailingStack())
        with pytest.raises(RuntimeError, match="second ContextVar set failed"):
            with observe_run(path, durable=False):
                pytest.fail("context body must not run")

    assert closed == [True]
    assert runtime_module._CURRENT_RUN.get() is None
    assert original_stack.get() == ()
    assert logger.handlers == original_handlers
    assert warnings.showwarning is original_showwarning
    assert runtime_module._BRIDGE_USERS == 0


def test_fsync_cleanup_error_preserves_success_and_application_error(
    tmp_path, monkeypatch
) -> None:
    success_path = tmp_path / "fsync-success.jsonl"
    failure_path = tmp_path / "fsync-failure.jsonl"
    success_path.touch()
    failure_path.touch()

    def fail_fsync(descriptor) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    success_reached = False
    with observe_run(success_path):
        success_reached = True
    assert success_reached

    original_error = RuntimeError("application failed")
    with pytest.raises(RuntimeError) as caught:
        with observe_run(failure_path):
            raise original_error
    assert caught.value is original_error

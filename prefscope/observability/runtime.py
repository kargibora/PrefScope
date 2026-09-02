"""Automatic, opt-in runtime recording for PrefScope operations."""

from __future__ import annotations

import atexit
import builtins
import logging
import os
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from prefscope.observability.events import RunEvent
from prefscope.observability.recorder import DEFAULT_MAX_EVENT_BYTES, JsonlRecorder

_ENVIRONMENT_PATH = "PREFSCOPE_EVENTS_PATH"
_ENVIRONMENT_PRETTY = "PREFSCOPE_EVENTS_PRETTY"
_ENVIRONMENT_TRUTHY = frozenset({"1", "true", "yes", "on"})
_PREFSCOPE_LOGGER = logging.getLogger("prefscope")
_CURRENT_RUN: ContextVar[RunContext | None] = ContextVar(
    "prefscope_observability_run", default=None
)
_OPERATION_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "prefscope_observability_operations", default=()
)


class RunContext:
    """The active recorder for an :func:`observe_run` context.

    ``record`` and ``record_failure`` expose the existing validated event
    contract. Automatic instrumentation uses the same recorder and never
    inspects a function's arguments or return value.
    """

    def __init__(
        self,
        recorder: JsonlRecorder,
        *,
        monotonic: Callable[[], float],
        reporter: Any | None = None,
    ) -> None:
        self._recorder = recorder
        self._monotonic = monotonic
        self._reporter = reporter
        self._reporter_closed = False

    @property
    def recorder(self) -> JsonlRecorder:
        """Return the underlying JSONL recorder."""
        return self._recorder

    @property
    def run_id(self) -> str:
        """Return the identifier shared by this run's events."""
        return self._recorder.run_id

    @property
    def path(self) -> os.PathLike[str]:
        """Return the event-stream path."""
        return self._recorder.path

    @property
    def closed(self) -> bool:
        """Report whether the underlying recorder is closed."""
        return self._recorder.closed

    def record(
        self,
        stage: str,
        status: str,
        message: str = "",
        data: Mapping[str, Any] | None = None,
    ) -> RunEvent:
        """Append one event using the bounded and redacted event schema."""
        event = self._recorder.record(stage, status, message, data)
        self._dispatch(event)
        return event

    def record_failure(
        self,
        stage: str,
        error: BaseException,
        *,
        message: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> RunEvent:
        """Append a sanitized failure without a traceback or exception arguments."""
        event = self._recorder.record_failure(
            stage,
            error,
            message=message,
            data=data,
        )
        self._dispatch(event)
        return event

    def _dispatch(self, event: RunEvent) -> None:
        reporter = self._reporter
        if reporter is None:
            return
        try:
            reporter.observe(event)
        except BaseException:
            # Presentation is best effort and cannot affect durable recording.
            pass

    def close(self) -> None:
        """Close the recorder and the optional reporter. Repeated calls are safe."""
        try:
            self._recorder.close()
        finally:
            if not self._reporter_closed:
                self._reporter_closed = True
                reporter = self._reporter
                if reporter is not None:
                    try:
                        reporter.close()
                    except BaseException:
                        # Presentation cleanup cannot alter recorder behavior.
                        pass


_ENVIRONMENT_RUN: RunContext | None = None
_ENVIRONMENT_LOCK = threading.RLock()
_BRIDGE_LOCK = threading.RLock()
_BRIDGE_USERS = 0
_PREVIOUS_SHOWWARNING: Callable[..., Any] | None = None


def _active_run() -> RunContext | None:
    active = _CURRENT_RUN.get()
    if active is not None:
        return None if active.closed else active
    with _ENVIRONMENT_LOCK:
        environment = _ENVIRONMENT_RUN
        if environment is not None and not environment.closed:
            return environment
    return None


def _operation_links() -> dict[str, str | None]:
    stack = _OPERATION_STACK.get()
    return {
        "operation_id": stack[-1] if stack else None,
        "parent_operation_id": stack[-2] if len(stack) > 1 else None,
    }


def _builtin_exception_name(value: object) -> str:
    """Return a fixed-safe built-in exception name or a generic fallback."""
    if not isinstance(value, type) or not issubclass(value, BaseException):
        return "Exception"
    name = value.__name__
    if value.__module__ != "builtins" or getattr(builtins, name, None) is not value:
        return "Exception"
    return name


def _builtin_warning_name(value: object) -> str:
    """Return a fixed-safe built-in warning name or a generic fallback."""
    if not isinstance(value, type) or not issubclass(value, Warning):
        return "Warning"
    name = value.__name__
    if value.__module__ != "builtins" or getattr(builtins, name, None) is not value:
        return "Warning"
    return name


class _ContextLoggingHandler(logging.Handler):
    """Dispatch PrefScope log records to the context-local recorder."""

    def emit(self, record: logging.LogRecord) -> None:
        run = _active_run()
        if run is None:
            return
        try:
            if record.levelno >= logging.ERROR:
                status = "failed"
            elif record.levelno >= logging.WARNING:
                status = "warning"
            else:
                status = "info"
            data: dict[str, Any] = {
                "logger": "prefscope",
                "level": (
                    "ERROR"
                    if record.levelno >= logging.ERROR
                    else "WARNING"
                    if record.levelno >= logging.WARNING
                    else "INFO"
                ),
                **_operation_links(),
            }
            if record.exc_info and record.exc_info[0] is not None:
                data["error_type"] = _builtin_exception_name(record.exc_info[0])
            # Formatted messages, arguments, exception text, tracebacks, and
            # stack text are deliberately excluded from automatic events.
            run.record("logging", status, data=data)
        except BaseException:
            # Automatic observation is best effort and must never affect the
            # application log call.
            pass


_CONTEXT_LOG_HANDLER = _ContextLoggingHandler()


def _showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,
    line: str | None = None,
) -> None:
    run = _active_run()
    if run is not None:
        try:
            category_name = _builtin_warning_name(category)
            # Warning text and source locations may contain prompts, row
            # values, or local paths. Store only its safe category.
            run.record(
                "warnings",
                "warning",
                data={"category": category_name, **_operation_links()},
            )
        except BaseException:
            # Warning capture is best effort and must not suppress forwarding.
            pass
    previous = _PREVIOUS_SHOWWARNING
    if previous is not None:
        previous(message, category, filename, lineno, file=file, line=line)


def _install_bridges() -> None:
    global _BRIDGE_USERS, _PREVIOUS_SHOWWARNING
    with _BRIDGE_LOCK:
        if _BRIDGE_USERS == 0:
            previous_warning = warnings.showwarning
            try:
                _PREFSCOPE_LOGGER.addHandler(_CONTEXT_LOG_HANDLER)
                _PREVIOUS_SHOWWARNING = previous_warning
                warnings.showwarning = _showwarning
            except BaseException:
                # Roll back exact process-global state even when a step failed
                # after partially mutating the logger or warning hook.
                try:
                    if _CONTEXT_LOG_HANDLER in _PREFSCOPE_LOGGER.handlers:
                        _PREFSCOPE_LOGGER.removeHandler(_CONTEXT_LOG_HANDLER)
                except BaseException:
                    try:
                        _PREFSCOPE_LOGGER.handlers.remove(_CONTEXT_LOG_HANDLER)
                    except BaseException:
                        pass
                try:
                    warnings.showwarning = previous_warning
                except BaseException:
                    pass
                _PREVIOUS_SHOWWARNING = None
                _BRIDGE_USERS = 0
                raise
        _BRIDGE_USERS += 1


def _remove_bridges() -> None:
    global _BRIDGE_USERS, _PREVIOUS_SHOWWARNING
    with _BRIDGE_LOCK:
        if _BRIDGE_USERS == 0:
            return
        _BRIDGE_USERS -= 1
        if _BRIDGE_USERS != 0:
            return
        first_error: BaseException | None = None
        try:
            _PREFSCOPE_LOGGER.removeHandler(_CONTEXT_LOG_HANDLER)
        except BaseException as error:
            first_error = error
            # ``logging.Logger.removeHandler`` only removes one list entry.
            # Make the same best-effort mutation if a patched method failed.
            try:
                _PREFSCOPE_LOGGER.handlers.remove(_CONTEXT_LOG_HANDLER)
            except BaseException:
                pass
        try:
            if (
                warnings.showwarning is _showwarning
                and _PREVIOUS_SHOWWARNING is not None
            ):
                warnings.showwarning = _PREVIOUS_SHOWWARNING
        except BaseException as error:
            if first_error is None:
                first_error = error
        finally:
            _PREVIOUS_SHOWWARNING = None
        if first_error is not None:
            raise first_error


def _install_bridges_guarded() -> None:
    """Install one bridge user and roll back a partial failed registration."""
    with _BRIDGE_LOCK:
        users_before = _BRIDGE_USERS
        try:
            _install_bridges()
        except BaseException:
            if _BRIDGE_USERS > users_before:
                try:
                    _remove_bridges()
                except BaseException:
                    pass
            raise


def _pretty_from_environment() -> bool:
    try:
        value = os.environ.get(_ENVIRONMENT_PRETTY, "")
        return type(value) is str and value.strip().lower() in _ENVIRONMENT_TRUTHY
    except BaseException:
        return False


def _create_console_reporter() -> Any | None:
    """Create the optional reporter without importing Rich on disabled paths."""
    try:
        from prefscope.observability.console import ConsoleReporter

        return ConsoleReporter()
    except BaseException:
        return None


@contextmanager
def observe_run(
    path: str | os.PathLike[str],
    *,
    pretty: bool | None = None,
    run_id: str | None = None,
    durable: bool = True,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
    monotonic: Callable[[], float] | None = None,
    utc_now: Callable[[], datetime] | None = None,
) -> Iterator[RunContext]:
    """Activate automatic observability within a context.

    The context owns and closes its recorder. Nested contexts route events to
    the innermost recorder and restore the outer context on exit. ``pretty=None``
    consults ``PREFSCOPE_EVENTS_PRETTY``; an explicit boolean overrides it.
    Only log records below the ``prefscope`` logger and emitted Python warnings
    are bridged.
    """
    if pretty is not None and not isinstance(pretty, bool):
        raise ValueError("pretty must be a bool or None")
    pretty_enabled = _pretty_from_environment() if pretty is None else pretty
    clock = time.monotonic if monotonic is None else monotonic
    recorder = JsonlRecorder(
        path,
        run_id=run_id,
        durable=durable,
        max_event_bytes=max_event_bytes,
        monotonic=clock,
        utc_now=utc_now,
    )
    reporter = _create_console_reporter() if pretty_enabled else None
    run = RunContext(recorder, monotonic=clock, reporter=reporter)
    bridge_installed = False
    run_token: object | None = None
    operations_token: object | None = None
    try:
        _install_bridges_guarded()
        bridge_installed = True
        run_token = _CURRENT_RUN.set(run)
        operations_token = _OPERATION_STACK.set(())
    except BaseException:
        # Entry did not complete. Undo every successful activation step and
        # preserve the original setup error.
        if operations_token is not None:
            try:
                _OPERATION_STACK.reset(operations_token)
            except BaseException:
                pass
        if run_token is not None:
            try:
                _CURRENT_RUN.reset(run_token)
            except BaseException:
                pass
        if bridge_installed:
            try:
                _remove_bridges()
            except BaseException:
                pass
        try:
            run.close()
        except BaseException:
            pass
        raise
    try:
        yield run
    finally:
        # After entry succeeds, observability cleanup is entirely best effort.
        # It must not turn success into failure or replace an application error.
        if operations_token is not None:
            try:
                _OPERATION_STACK.reset(operations_token)
            except BaseException:
                pass
        if run_token is not None:
            try:
                _CURRENT_RUN.reset(run_token)
            except BaseException:
                pass
        if bridge_installed:
            try:
                _remove_bridges()
            except BaseException:
                pass
        try:
            run.close()
        except BaseException:
            pass


def _environment_run() -> RunContext | None:
    """Create the environment-selected recorder on first automatic use."""
    global _ENVIRONMENT_RUN
    with _ENVIRONMENT_LOCK:
        if _ENVIRONMENT_RUN is not None:
            if not _ENVIRONMENT_RUN.closed:
                return _ENVIRONMENT_RUN
            _ENVIRONMENT_RUN = None
        path = os.environ.get(_ENVIRONMENT_PATH)
        if not path:
            return None
        clock = time.monotonic
        try:
            recorder = JsonlRecorder(path, monotonic=clock)
        except BaseException:
            # An opt-in environment variable must not make instrumented code
            # fail when its observation destination is unavailable or unsafe.
            return None
        reporter = _create_console_reporter() if _pretty_from_environment() else None
        run = RunContext(recorder, monotonic=clock, reporter=reporter)
        _ENVIRONMENT_RUN = run
        try:
            _install_bridges_guarded()
        except BaseException:
            # Lazy environment activation is automatic and must never affect
            # the instrumented operation, even after a partial bridge install.
            _ENVIRONMENT_RUN = None
            try:
                run.close()
            except BaseException:
                pass
            return None
        return run


def _close_environment_recorder() -> None:
    """Close and forget the lazily created environment recorder."""
    global _ENVIRONMENT_RUN
    with _ENVIRONMENT_LOCK:
        run = _ENVIRONMENT_RUN
        if run is None:
            return
        _ENVIRONMENT_RUN = None
        try:
            run.close()
        except BaseException:
            pass
        try:
            _remove_bridges()
        except BaseException:
            pass


class _OperationSpan:
    """Completion data supplied explicitly by an instrumented call site."""

    def __init__(self, *, active: bool) -> None:
        self._active = active
        self._result_data: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        """Report whether this span has an active recorder."""
        return self._active

    def update(self, **safe_data: Any) -> None:
        """Add explicit structured fields to the eventual completion event."""
        self._result_data.update(safe_data)

    def set_result_data(self, data: Mapping[str, Any]) -> None:
        """Replace completion fields with an explicit structured mapping."""
        if not isinstance(data, Mapping):
            raise ValueError("operation result data must be an object")
        self._result_data = dict(data)


@contextmanager
def automatic_stage(
    stage: str,
    data: Mapping[str, Any] | None = None,
) -> Iterator[_OperationSpan]:
    """Record one automatically instrumented operation when recording is active.

    This is an internal instrumentation hook. Callers may supply only explicit
    structured event data. The hook never inspects or serializes function
    arguments, return values, locals, or tracebacks. The yielded span accepts
    explicit output counts or shapes for the completion event; :class:`RunEvent`
    applies the normal bounds and redaction when that event is recorded.
    """
    explicit_run = _CURRENT_RUN.get()
    if explicit_run is not None:
        run = None if explicit_run.closed else explicit_run
    else:
        run = _environment_run()
    span = _OperationSpan(active=run is not None)
    if run is None:
        yield span
        return

    try:
        operation_id = str(uuid.uuid4())
    except BaseException:
        # Correlation is mandatory for automatic events. If ID generation is
        # unavailable, execute unchanged with an explicitly inactive span.
        yield _OperationSpan(active=False)
        return
    try:
        stack = _OPERATION_STACK.get()
        token = _OPERATION_STACK.set((*stack, operation_id))
    except BaseException:
        yield _OperationSpan(active=False)
        return
    parent_operation_id = stack[-1] if stack else None
    try:
        base_data = dict(data or {})
    except BaseException:
        # A hostile or malformed Mapping must not prevent the operation.
        base_data = {}
    base_data["operation_id"] = operation_id
    base_data["parent_operation_id"] = parent_operation_id
    try:
        started_at = run._monotonic()
    except BaseException:
        started_at = 0.0
    try:
        run.record(stage, "started", data=base_data)
    except BaseException:
        pass
    try:
        yield span
    except BaseException as error:
        try:
            failure_data = dict(base_data)
            try:
                duration = max(0.0, run._monotonic() - started_at)
            except BaseException:
                duration = 0.0
            failure_data["duration_seconds"] = duration
            # Automatic failure events deliberately omit ``str(error)``. An
            # exception message can contain prompts, row values, or local paths.
            failure_data["error_type"] = _builtin_exception_name(type(error))
            run.record(
                stage,
                "failed",
                message=f"{stage} failed",
                data=failure_data,
            )
        except BaseException:
            # Never replace the operation's exception with an observation error.
            pass
        raise
    else:
        try:
            completed_data = dict(base_data)
            completed_data.update(span._result_data)
            # Correlation and timing fields cannot be replaced by caller data.
            completed_data["operation_id"] = operation_id
            completed_data["parent_operation_id"] = parent_operation_id
            try:
                duration = max(0.0, run._monotonic() - started_at)
            except BaseException:
                duration = 0.0
            completed_data["duration_seconds"] = duration
            run.record(stage, "completed", data=completed_data)
        except BaseException:
            # Automatic observation must not turn success into failure.
            pass
    finally:
        try:
            _OPERATION_STACK.reset(token)
        except BaseException:
            pass


atexit.register(_close_environment_recorder)

__all__ = ["RunContext", "observe_run"]

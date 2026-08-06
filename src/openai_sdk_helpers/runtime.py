"""Typed operation lifecycle and vendor-neutral observability hooks."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeVar, cast

T = TypeVar("T")

_DEFAULT_REDACTED_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "content",
        "file",
        "input",
        "output",
        "password",
        "prompt",
        "response",
        "secret",
        "token",
        "tool_arguments",
    }
)


class OperationPhase(str, Enum):
    """Lifecycle phase emitted to operation observers."""

    START = "start"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class OperationUsage:
    """Optional token usage captured from an underlying SDK result.

    Parameters
    ----------
    input_tokens : int or None, default=None
        Input token count reported by the SDK.
    output_tokens : int or None, default=None
        Output token count reported by the SDK.
    total_tokens : int or None, default=None
        Total token count reported by the SDK.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_value(cls, value: object) -> OperationUsage | None:
        """Build usage information from an SDK object or mapping.

        Parameters
        ----------
        value : object
            SDK usage object or mapping.

        Returns
        -------
        OperationUsage or None
            Parsed usage when at least one supported field is present.
        """
        if value is None:
            return None
        input_tokens = _read_value(value, "input_tokens")
        output_tokens = _read_value(value, "output_tokens")
        total_tokens = _read_value(value, "total_tokens")
        parsed = cls(
            input_tokens=_optional_int(input_tokens),
            output_tokens=_optional_int(output_tokens),
            total_tokens=_optional_int(total_tokens),
        )
        if all(
            token_count is None
            for token_count in (
                parsed.input_tokens,
                parsed.output_tokens,
                parsed.total_tokens,
            )
        ):
            return None
        return parsed


@dataclass(frozen=True, slots=True)
class OperationEvent:
    """One lifecycle event delivered to an operation observer."""

    phase: OperationPhase
    operation_name: str
    request_id: str | None
    correlation_id: str | None
    trace_id: str | None
    model: str | None
    started_at: float
    finished_at: float | None
    duration_seconds: float | None
    retry_count: int
    usage: OperationUsage | None
    metadata: Mapping[str, object]
    result: object | None = field(default=None, repr=False)
    error: BaseException | None = field(default=None, repr=False)

    def diagnostics(
        self,
        *,
        redact_keys: Iterable[str] | None = None,
        include_sensitive: bool = False,
    ) -> dict[str, object]:
        """Return safe diagnostic data without prompt or response content.

        Parameters
        ----------
        redact_keys : Iterable[str] or None, default=None
            Additional metadata keys to redact case-insensitively.
        include_sensitive : bool, default=False
            Include metadata values otherwise covered by the default redaction
            set. Raw results and exception messages are never serialized.

        Returns
        -------
        dict[str, object]
            JSON-compatible diagnostic fields.
        """
        keys = set() if include_sensitive else set(_DEFAULT_REDACTED_KEYS)
        if redact_keys is not None:
            keys.update(key.lower() for key in redact_keys)
        metadata = {
            key: "<redacted>" if _should_redact(key, keys) else value
            for key, value in self.metadata.items()
        }
        diagnostics: dict[str, object] = {
            "phase": self.phase.value,
            "operation_name": self.operation_name,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
            "metadata": metadata,
        }
        if self.usage is not None:
            diagnostics["usage"] = {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        if self.error is not None:
            diagnostics["error_type"] = type(self.error).__name__
        return diagnostics


class OperationObserver(Protocol):
    """Callable receiving operation lifecycle events inline."""

    def __call__(self, event: OperationEvent) -> None:
        """Receive one operation lifecycle event."""
        ...


@dataclass(slots=True)
class OperationContext:
    """Explicit, per-operation metadata and lifecycle state.

    A context is optional and represents exactly one operation. It is not safe
    to reuse one instance concurrently. Observer callbacks run inline; shared
    observers are responsible for their own thread safety.

    Parameters
    ----------
    operation_name : str
        Stable operation identifier such as ``responses.run_sync``.
    request_id : str or None, default=None
        Caller or SDK request identifier.
    correlation_id : str or None, default=None
        Identifier shared across related operations.
    trace_id : str or None, default=None
        Caller-managed trace identifier. This does not replace Agents tracing.
    model : str or None, default=None
        Model name when known.
    metadata : Mapping[str, object], default={}
        Caller metadata copied at construction time.
    observers : tuple[OperationObserver, ...], default=()
        Inline lifecycle observers.
    retry_count : int, default=0
        Retry attempts already performed by the caller or SDK wrapper.
    """

    operation_name: str
    request_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    model: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    observers: tuple[OperationObserver, ...] = ()
    retry_count: int = 0
    usage: OperationUsage | None = field(default=None, init=False)
    started_at: float | None = field(default=None, init=False)
    finished_at: float | None = field(default=None, init=False)
    observer_error_count: int = field(default=0, init=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize immutable inputs and validate lifecycle counters."""
        name = self.operation_name.strip()
        if not name:
            raise ValueError("operation_name must not be empty")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        self.operation_name = name
        self.metadata = MappingProxyType(dict(self.metadata))
        self.observers = tuple(self.observers)

    def record_retry(self, count: int = 1) -> None:
        """Increment the explicit retry count.

        Parameters
        ----------
        count : int, default=1
            Positive number of retries to add.
        """
        if count < 1:
            raise ValueError("count must be positive")
        self.retry_count += count

    def start(self) -> OperationEvent:
        """Start the operation and emit a start event.

        Returns
        -------
        OperationEvent
            Emitted start event.
        """
        if self.started_at is not None:
            raise RuntimeError("OperationContext has already started")
        self.started_at = time.time()
        event = self._event(OperationPhase.START)
        self._emit(event)
        return event

    def succeed(self, result: object) -> OperationEvent:
        """Finish successfully and emit the unmodified result to observers.

        Parameters
        ----------
        result : object
            Original SDK or handler result.

        Returns
        -------
        OperationEvent
            Emitted success event.
        """
        self._prepare_finish()
        self._capture_result_metadata(result)
        event = self._event(OperationPhase.SUCCESS, result=result)
        self._emit(event)
        return event

    def fail(self, error: BaseException) -> OperationEvent:
        """Finish with failure and emit the original exception to observers.

        Parameters
        ----------
        error : BaseException
            Original exception, re-raised by execution helpers unchanged.

        Returns
        -------
        OperationEvent
            Emitted failure event.
        """
        self._prepare_finish()
        event = self._event(OperationPhase.FAILURE, error=error)
        self._emit(event)
        return event

    def _prepare_finish(self) -> None:
        if self.started_at is None:
            raise RuntimeError("OperationContext must be started before finishing")
        if self._finished:
            raise RuntimeError("OperationContext has already finished")
        self.finished_at = time.time()
        self._finished = True

    def _capture_result_metadata(self, result: object) -> None:
        usage_value = _read_value(result, "usage")
        self.usage = OperationUsage.from_value(usage_value)
        if self.model is None:
            model = _read_value(result, "model")
            if isinstance(model, str):
                self.model = model
        if self.request_id is None:
            request_id = _read_value(result, "request_id")
            if request_id is None:
                request_id = _read_value(result, "_request_id")
            if isinstance(request_id, str):
                self.request_id = request_id

    def _event(
        self,
        phase: OperationPhase,
        *,
        result: object | None = None,
        error: BaseException | None = None,
    ) -> OperationEvent:
        if self.started_at is None:
            raise RuntimeError("OperationContext has not started")
        duration = None
        if self.finished_at is not None:
            duration = max(0.0, self.finished_at - self.started_at)
        return OperationEvent(
            phase=phase,
            operation_name=self.operation_name,
            request_id=self.request_id,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            model=self.model,
            started_at=self.started_at,
            finished_at=self.finished_at,
            duration_seconds=duration,
            retry_count=self.retry_count,
            usage=self.usage,
            metadata=self.metadata,
            result=result,
            error=error,
        )

    def _emit(self, event: OperationEvent) -> None:
        for observer in self.observers:
            try:
                observer(event)
            except Exception:
                self.observer_error_count += 1


def run_observed_sync(
    context: OperationContext | None,
    function: Callable[[], T],
) -> T:
    """Execute a callable with synchronous lifecycle events.

    Awaitable return values remain awaitable and emit completion only after the
    caller awaits them.

    Parameters
    ----------
    context : OperationContext or None
        Optional operation context.
    function : Callable[[], T]
        Operation callable.

    Returns
    -------
    T
        Original result without wrapping or conversion.
    """
    if context is None:
        return function()
    context.start()
    try:
        result = function()
    except BaseException as error:
        context.fail(error)
        raise
    if inspect.isawaitable(result):
        return cast(T, _finish_awaitable(context, cast(Awaitable[Any], result)))
    context.succeed(result)
    return result


async def run_observed_async(
    context: OperationContext | None,
    function: Callable[[], T | Awaitable[T]],
) -> T:
    """Execute a callable with async-compatible lifecycle events.

    Parameters
    ----------
    context : OperationContext or None
        Optional operation context.
    function : Callable[[], T or Awaitable[T]]
        Operation callable.

    Returns
    -------
    T
        Original resolved result without wrapping or conversion.
    """
    if context is None:
        result = function()
        if inspect.isawaitable(result):
            return await cast(Awaitable[T], result)
        return cast(T, result)
    context.start()
    try:
        result = function()
        if inspect.isawaitable(result):
            resolved = await cast(Awaitable[T], result)
        else:
            resolved = cast(T, result)
    except BaseException as error:
        context.fail(error)
        raise
    context.succeed(resolved)
    return resolved


async def _finish_awaitable(
    context: OperationContext,
    awaitable: Awaitable[T],
) -> T:
    try:
        result = await awaitable
    except BaseException as error:
        context.fail(error)
        raise
    context.succeed(result)
    return result


def _read_value(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _should_redact(key: str, redacted_keys: set[str]) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in redacted_keys)


__all__ = [
    "OperationContext",
    "OperationEvent",
    "OperationObserver",
    "OperationPhase",
    "OperationUsage",
    "run_observed_async",
    "run_observed_sync",
]

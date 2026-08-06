"""Realtime event normalization, explicit tool execution, and controls."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias, cast

from openai_sdk_helpers.runtime import OperationContext, run_observed_async


class RealtimeEventKind(str, Enum):
    """Normalized high-level kinds for official Realtime events."""

    SESSION = "session"
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INTERRUPTION = "interruption"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RealtimeEventEnvelope:
    """Normalized Realtime event preserving the original SDK payload.

    Parameters
    ----------
    type : str
        Original event type string.
    kind : RealtimeEventKind
        Stable high-level classification.
    item_id : str or None, default=None
        Conversation item identifier when present.
    response_id : str or None, default=None
        Response identifier when present.
    call_id : str or None, default=None
        Tool call identifier when present.
    name : str or None, default=None
        Tool or event name when present.
    text : str or None, default=None
        Text or transcript content when present.
    delta : str or bytes or None, default=None
        Incremental event payload when present.
    raw : object or None, default=None
        Original official SDK event.
    """

    type: str
    kind: RealtimeEventKind
    item_id: str | None = None
    response_id: str | None = None
    call_id: str | None = None
    name: str | None = None
    text: str | None = None
    delta: str | bytes | None = field(default=None, repr=False)
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize required and optional string identifiers."""
        object.__setattr__(self, "type", _required(self.type, "type"))
        for name in ("item_id", "response_id", "call_id", "name", "text"):
            value = getattr(self, name)
            if value is not None:
                normalized = value.strip()
                object.__setattr__(self, name, normalized or None)


class RealtimeEventNormalizationError(ValueError):
    """Malformed event data that cannot be normalized strictly."""


@dataclass(frozen=True, slots=True)
class RealtimeToolCall:
    """Normalized Realtime tool call with parsed arguments and raw access."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(repr=False)
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize identity and copy parsed arguments."""
        object.__setattr__(self, "call_id", _required(self.call_id, "call_id"))
        object.__setattr__(self, "name", _required(self.name, "name"))
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True, slots=True)
class RealtimeToolResult:
    """Explicit local result for one Realtime tool call."""

    call_id: str
    name: str
    output: object | None = field(default=None, repr=False)
    error: BaseException | None = field(default=None, repr=False, compare=False)
    raw_call: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize identity and reject ambiguous success/error results."""
        object.__setattr__(self, "call_id", _required(self.call_id, "call_id"))
        object.__setattr__(self, "name", _required(self.name, "name"))
        if self.output is not None and self.error is not None:
            raise ValueError("tool results cannot contain both output and error")

    @property
    def succeeded(self) -> bool:
        """Return whether execution completed without a captured error."""
        return self.error is None

    def serialized_output(self) -> str:
        """Serialize output for an official tool-output submission method."""
        if self.error is not None:
            raise RuntimeError("Cannot serialize a failed tool result") from self.error
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, default=str, separators=(",", ":"))


RealtimeToolHandler: TypeAlias = Callable[
    [Mapping[str, Any]],
    object | Awaitable[object],
]


class RealtimeToolRegistry:
    """Explicit local registry for Realtime tool handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, RealtimeToolHandler] = {}

    def register(
        self,
        name: str,
        handler: RealtimeToolHandler,
        *,
        replace: bool = False,
    ) -> None:
        """Register one handler with deterministic duplicate behavior."""
        normalized = _required(name, "name")
        if normalized in self._handlers and not replace:
            raise ValueError(f"Realtime tool {normalized!r} is already registered")
        self._handlers[normalized] = handler

    def unregister(self, name: str) -> bool:
        """Remove one handler and report whether it existed."""
        return self._handlers.pop(_required(name, "name"), None) is not None

    def list_names(self) -> tuple[str, ...]:
        """Return registered tool names in insertion order."""
        return tuple(self._handlers)

    async def execute(
        self,
        call: RealtimeToolCall,
        *,
        capture_errors: bool = False,
        operation_context: OperationContext | None = None,
    ) -> RealtimeToolResult:
        """Execute one registered handler without hidden retry or submission."""
        handler = self._handlers.get(call.name)
        if handler is None:
            raise KeyError(f"No Realtime handler registered for {call.name!r}")

        async def execute_handler() -> RealtimeToolResult:
            try:
                output = handler(call.arguments)
                if inspect.isawaitable(output):
                    output = await cast(Awaitable[object], output)
                return RealtimeToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    output=output,
                    raw_call=call.raw,
                )
            except BaseException as error:
                if not capture_errors:
                    raise
                return RealtimeToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    error=error,
                    raw_call=call.raw,
                )

        return await run_observed_async(operation_context, execute_handler)


class RealtimeEventSource(Protocol):
    """Asynchronous source of official Realtime events."""

    def __aiter__(self) -> AsyncIterator[object]:
        """Return the event iterator."""
        ...


class RealtimeControlProtocol(Protocol):
    """Explicit session controls used by local adapters and test transport."""

    async def interrupt(self) -> None:
        """Interrupt current generation or playback."""
        ...

    async def cancel(self) -> None:
        """Cancel the current response."""
        ...

    async def send_message(self, message: str) -> None:
        """Send one text message."""
        ...

    async def send_audio(self, audio: bytes) -> None:
        """Send one audio chunk."""
        ...

    async def send_tool_output(self, call_id: str, output: str) -> None:
        """Submit one serialized tool output."""
        ...


async def iter_realtime_events(
    source: RealtimeEventSource,
    *,
    strict: bool = True,
) -> AsyncIterator[RealtimeEventEnvelope]:
    """Yield normalized events while preserving raw source order."""
    async for raw_event in source:
        try:
            yield normalize_realtime_event(raw_event)
        except (TypeError, ValueError) as error:
            if strict:
                raise RealtimeEventNormalizationError(str(error)) from error
            event_type = _read(raw_event, "type") or "unknown"
            yield RealtimeEventEnvelope(
                type=str(event_type),
                kind=RealtimeEventKind.UNKNOWN,
                raw=raw_event,
            )


def normalize_realtime_event(raw_event: object) -> RealtimeEventEnvelope:
    """Normalize common official Realtime fields without discarding raw data."""
    event_type = _read(raw_event, "type")
    if not event_type:
        raise ValueError("Realtime event is missing type")
    item = _read(raw_event, "item")
    response = _read(raw_event, "response")
    item_id = _first(
        _read(raw_event, "item_id"),
        _read(item, "id"),
    )
    response_id = _first(
        _read(raw_event, "response_id"),
        _read(response, "id"),
    )
    call_id = _first(
        _read(raw_event, "call_id"),
        _read(item, "call_id"),
    )
    name = _first(
        _read(raw_event, "name"),
        _read(item, "name"),
    )
    text = _first(
        _read(raw_event, "transcript"),
        _read(raw_event, "text"),
        _read(item, "transcript"),
    )
    delta = _read(raw_event, "delta")
    if delta is not None and not isinstance(delta, (str, bytes)):
        delta = str(delta)
    return RealtimeEventEnvelope(
        type=str(event_type),
        kind=_event_kind(str(event_type)),
        item_id=str(item_id) if item_id else None,
        response_id=str(response_id) if response_id else None,
        call_id=str(call_id) if call_id else None,
        name=str(name) if name else None,
        text=str(text) if text else None,
        delta=cast(str | bytes | None, delta),
        raw=raw_event,
    )


def parse_realtime_tool_call(raw_event: object) -> RealtimeToolCall:
    """Parse one function/tool call event into a local execution contract."""
    item = _read(raw_event, "item")
    call_id = _first(_read(raw_event, "call_id"), _read(item, "call_id"))
    name = _first(_read(raw_event, "name"), _read(item, "name"))
    raw_arguments = _first(
        _read(raw_event, "arguments"),
        _read(item, "arguments"),
    )
    if not call_id:
        raise RealtimeEventNormalizationError("Realtime tool call is missing call_id")
    if not name:
        raise RealtimeEventNormalizationError("Realtime tool call is missing name")
    if raw_arguments is None:
        arguments: Mapping[str, Any] = {}
    elif isinstance(raw_arguments, Mapping):
        arguments = dict(raw_arguments)
    elif isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            raise RealtimeEventNormalizationError(
                "Realtime tool arguments are not valid JSON"
            ) from error
        if not isinstance(parsed, Mapping):
            raise RealtimeEventNormalizationError(
                "Realtime tool arguments must decode to an object"
            )
        arguments = dict(parsed)
    else:
        raise RealtimeEventNormalizationError(
            "Realtime tool arguments must be a mapping or JSON object string"
        )
    return RealtimeToolCall(
        call_id=str(call_id),
        name=str(name),
        arguments=arguments,
        raw=raw_event,
    )


async def interrupt_realtime_session(
    session: object,
    *,
    operation_context: OperationContext | None = None,
) -> None:
    """Invoke the session's explicit interruption method."""
    await _invoke_control(session, "interrupt", operation_context=operation_context)


async def cancel_realtime_response(
    session: object,
    *,
    operation_context: OperationContext | None = None,
) -> None:
    """Invoke an explicit response cancellation method."""
    method_name = "cancel_response" if hasattr(session, "cancel_response") else "cancel"
    await _invoke_control(session, method_name, operation_context=operation_context)


async def send_realtime_message(
    session: object,
    message: str,
    *,
    operation_context: OperationContext | None = None,
) -> None:
    """Send one non-empty text message through the raw session."""
    await _invoke_control(
        session,
        "send_message",
        _required(message, "message"),
        operation_context=operation_context,
    )


async def send_realtime_audio(
    session: object,
    audio: bytes,
    *,
    operation_context: OperationContext | None = None,
) -> None:
    """Send one non-empty audio chunk without managing an audio device."""
    if not audio:
        raise ValueError("audio must not be empty")
    await _invoke_control(
        session,
        "send_audio",
        audio,
        operation_context=operation_context,
    )


async def submit_realtime_tool_result(
    session: object,
    result: RealtimeToolResult,
    *,
    operation_context: OperationContext | None = None,
) -> None:
    """Submit one successful local tool result explicitly."""
    await _invoke_control(
        session,
        "send_tool_output",
        result.call_id,
        result.serialized_output(),
        operation_context=operation_context,
    )


async def _invoke_control(
    session: object,
    method_name: str,
    *args: object,
    operation_context: OperationContext | None,
) -> None:
    method = getattr(session, method_name, None)
    if method is None or not callable(method):
        raise TypeError(f"Realtime session does not support {method_name}()")

    async def execute() -> None:
        value = method(*args)
        if inspect.isawaitable(value):
            await cast(Awaitable[object], value)

    await run_observed_async(operation_context, execute)


def _event_kind(event_type: str) -> RealtimeEventKind:
    lowered = event_type.lower()
    if "error" in lowered or "failed" in lowered:
        return RealtimeEventKind.ERROR
    if "function_call" in lowered or "tool_call" in lowered:
        if "output" in lowered or "result" in lowered:
            return RealtimeEventKind.TOOL_RESULT
        return RealtimeEventKind.TOOL_CALL
    if "interrupt" in lowered or "cancel" in lowered or "truncat" in lowered:
        return RealtimeEventKind.INTERRUPTION
    if "transcript" in lowered or "text" in lowered:
        return RealtimeEventKind.TRANSCRIPT
    if "audio" in lowered:
        return RealtimeEventKind.AUDIO
    if "session" in lowered or "response" in lowered or "conversation" in lowered:
        return RealtimeEventKind.SESSION
    return RealtimeEventKind.UNKNOWN


def _first(*values: object | None) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _read(value: object | None, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


__all__ = [
    "RealtimeControlProtocol",
    "RealtimeEventEnvelope",
    "RealtimeEventKind",
    "RealtimeEventNormalizationError",
    "RealtimeEventSource",
    "RealtimeToolCall",
    "RealtimeToolHandler",
    "RealtimeToolRegistry",
    "RealtimeToolResult",
    "cancel_realtime_response",
    "interrupt_realtime_session",
    "iter_realtime_events",
    "normalize_realtime_event",
    "parse_realtime_tool_call",
    "send_realtime_audio",
    "send_realtime_message",
    "submit_realtime_tool_result",
]

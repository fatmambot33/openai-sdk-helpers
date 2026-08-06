"""Sequential Realtime event consumption and explicit tool-call policy."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias, cast

from openai_sdk_helpers.runtime import OperationContext, run_observed_async

from .events import (
    RealtimeEventEnvelope,
    RealtimeEventSource,
    RealtimeToolCall,
    RealtimeToolRegistry,
    RealtimeToolResult,
    iter_realtime_events,
    parse_realtime_tool_call,
    submit_realtime_tool_result,
)

RealtimeEventCallback: TypeAlias = Callable[
    [RealtimeEventEnvelope],
    object | Awaitable[object],
]


class RealtimeToolApprovalDecision(str, Enum):
    """Explicit application decision for one Realtime tool call."""

    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class RealtimeToolApprovalRequest:
    """Approval request that hides tool arguments from its representation."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(repr=False)
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize identity and copy caller-owned arguments."""
        call_id = self.call_id.strip()
        name = self.name.strip()
        if not call_id:
            raise ValueError("call_id must not be empty")
        if not name:
            raise ValueError("name must not be empty")
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", dict(self.arguments))


class RealtimeToolApprovalHandler(Protocol):
    """Application-owned sync or async approval callback."""

    def __call__(
        self,
        request: RealtimeToolApprovalRequest,
    ) -> RealtimeToolApprovalDecision | Awaitable[RealtimeToolApprovalDecision]:
        """Approve or reject one tool call explicitly."""
        ...


async def consume_realtime_events(
    source: RealtimeEventSource,
    callback: RealtimeEventCallback,
    *,
    strict: bool = True,
) -> int:
    """Consume normalized events sequentially through one callback.

    Callback invocation follows source order. Callback exceptions stop
    consumption and are propagated unchanged; no background task or duplicate
    event state machine is created.
    """
    count = 0
    async for event in iter_realtime_events(source, strict=strict):
        value = callback(event)
        if inspect.isawaitable(value):
            await cast(Awaitable[object], value)
        count += 1
    return count


async def request_realtime_tool_approval(
    call: RealtimeToolCall,
    handler: RealtimeToolApprovalHandler | None,
) -> RealtimeToolApprovalDecision:
    """Resolve one tool approval with a fail-closed default."""
    if handler is None:
        return RealtimeToolApprovalDecision.REJECT
    request = RealtimeToolApprovalRequest(
        call_id=call.call_id,
        name=call.name,
        arguments=call.arguments,
        raw=call.raw,
    )
    decision = handler(request)
    if inspect.isawaitable(decision):
        decision = await cast(Awaitable[RealtimeToolApprovalDecision], decision)
    if not isinstance(decision, RealtimeToolApprovalDecision):
        return RealtimeToolApprovalDecision.REJECT
    return decision


async def execute_realtime_tool_call(
    registry: RealtimeToolRegistry,
    call: RealtimeToolCall,
    *,
    approval_handler: RealtimeToolApprovalHandler | None = None,
    require_approval: bool = True,
    timeout_seconds: float | None = None,
    capture_errors: bool = False,
    operation_context: OperationContext | None = None,
) -> RealtimeToolResult:
    """Approve and execute one registered tool without hidden submission.

    Unknown tools are rejected by the registry. Approval defaults to rejection.
    Timeouts and cancellations stop the underlying awaitable. No retry occurs.
    When ``capture_errors`` is true, rejection, timeout, and handler failures are
    returned in ``RealtimeToolResult.error``; otherwise they are raised.
    """
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    async def execute() -> RealtimeToolResult:
        if require_approval:
            decision = await request_realtime_tool_approval(call, approval_handler)
            if decision is not RealtimeToolApprovalDecision.APPROVE:
                error = PermissionError(
                    f"Realtime tool {call.name!r} was not explicitly approved"
                )
                if capture_errors:
                    return RealtimeToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        error=error,
                        raw_call=call.raw,
                    )
                raise error
        coroutine = registry.execute(
            call,
            capture_errors=capture_errors,
            operation_context=operation_context,
        )
        if timeout_seconds is None:
            return await coroutine
        try:
            return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
        except TimeoutError as error:
            if capture_errors:
                return RealtimeToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    error=error,
                    raw_call=call.raw,
                )
            raise

    return await run_observed_async(operation_context, execute)


async def process_realtime_tool_event(
    session: object,
    registry: RealtimeToolRegistry,
    raw_event: object,
    *,
    approval_handler: RealtimeToolApprovalHandler | None = None,
    require_approval: bool = True,
    timeout_seconds: float | None = None,
    operation_context: OperationContext | None = None,
) -> RealtimeToolResult:
    """Parse, approve, execute, and explicitly submit one tool event."""
    call = parse_realtime_tool_call(raw_event)
    result = await execute_realtime_tool_call(
        registry,
        call,
        approval_handler=approval_handler,
        require_approval=require_approval,
        timeout_seconds=timeout_seconds,
        operation_context=operation_context,
    )
    await submit_realtime_tool_result(
        session,
        result,
        operation_context=operation_context,
    )
    return result


__all__ = [
    "RealtimeEventCallback",
    "RealtimeToolApprovalDecision",
    "RealtimeToolApprovalHandler",
    "RealtimeToolApprovalRequest",
    "consume_realtime_events",
    "execute_realtime_tool_call",
    "process_realtime_tool_event",
    "request_realtime_tool_approval",
]

"""Deterministic tests for Realtime event, tool, and control helpers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openai_sdk_helpers.realtime import (
    InMemoryRealtimeSession,
    RealtimeEventKind,
    RealtimeEventNormalizationError,
    RealtimeToolApprovalDecision,
    RealtimeToolApprovalRequest,
    RealtimeToolCall,
    RealtimeToolRegistry,
    RealtimeToolResult,
    cancel_realtime_response,
    consume_realtime_events,
    execute_realtime_tool_call,
    interrupt_realtime_session,
    iter_realtime_events,
    normalize_realtime_event,
    parse_realtime_tool_call,
    process_realtime_tool_event,
    request_realtime_tool_approval,
    send_realtime_audio,
    send_realtime_message,
    submit_realtime_tool_result,
)


def test_normalize_event_preserves_common_fields_and_raw_object() -> None:
    raw = {
        "type": "response.audio_transcript.delta",
        "item_id": "item_1",
        "response_id": "resp_1",
        "delta": "hello",
    }

    event = normalize_realtime_event(raw)

    assert event.kind is RealtimeEventKind.TRANSCRIPT
    assert event.item_id == "item_1"
    assert event.response_id == "resp_1"
    assert event.delta == "hello"
    assert event.raw is raw


@pytest.mark.asyncio
async def test_event_iteration_preserves_order_and_supports_lenient_data() -> None:
    session = InMemoryRealtimeSession()
    valid = {"type": "response.audio.delta", "delta": b"audio"}
    malformed = {"delta": "missing type"}
    await session.push_event(valid)
    await session.push_event(malformed)
    await session.finish()

    events = [event async for event in iter_realtime_events(session, strict=False)]

    assert [event.kind for event in events] == [
        RealtimeEventKind.AUDIO,
        RealtimeEventKind.UNKNOWN,
    ]
    assert events[0].raw is valid
    assert events[1].raw is malformed


@pytest.mark.asyncio
async def test_strict_event_iteration_rejects_malformed_data() -> None:
    session = InMemoryRealtimeSession()
    await session.push_event({"delta": "missing type"})
    await session.finish()

    with pytest.raises(RealtimeEventNormalizationError):
        _ = [event async for event in iter_realtime_events(session)]


@pytest.mark.asyncio
async def test_callback_consumption_is_sequential_and_counts_events() -> None:
    session = InMemoryRealtimeSession()
    await session.push_event({"type": "session.created"})
    await session.push_event({"type": "response.audio.done"})
    await session.finish()
    seen: list[str] = []

    async def callback(event: Any) -> None:
        await asyncio.sleep(0)
        seen.append(event.type)

    count = await consume_realtime_events(session, callback)

    assert count == 2
    assert seen == ["session.created", "response.audio.done"]


@pytest.mark.asyncio
async def test_callback_failure_stops_consumption_without_background_tasks() -> None:
    session = InMemoryRealtimeSession()
    await session.push_event({"type": "session.created"})
    await session.push_event({"type": "session.updated"})
    await session.finish()
    seen: list[str] = []

    def callback(event: Any) -> None:
        seen.append(event.type)
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        await consume_realtime_events(session, callback)

    assert seen == ["session.created"]


def test_parse_tool_call_accepts_mapping_and_json_arguments() -> None:
    mapping_raw = {
        "type": "response.function_call_arguments.done",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": {"query": "docs"},
    }
    json_raw = {
        "type": "response.function_call_arguments.done",
        "item": {
            "call_id": "call_2",
            "name": "lookup",
            "arguments": '{"query":"api"}',
        },
    }

    mapping_call = parse_realtime_tool_call(mapping_raw)
    json_call = parse_realtime_tool_call(json_raw)

    assert mapping_call.arguments == {"query": "docs"}
    assert json_call.arguments == {"query": "api"}
    assert mapping_call.raw is mapping_raw
    assert json_call.raw is json_raw


def test_parse_tool_call_rejects_non_object_arguments() -> None:
    with pytest.raises(RealtimeEventNormalizationError, match="decode to an object"):
        parse_realtime_tool_call(
            {
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "[]",
            }
        )


@pytest.mark.asyncio
async def test_tool_registry_executes_sync_and_async_handlers() -> None:
    registry = RealtimeToolRegistry()
    registry.register("add", lambda arguments: arguments["left"] + arguments["right"])

    async def echo(arguments: Any) -> Any:
        await asyncio.sleep(0)
        return arguments

    registry.register("echo", echo)

    added = await registry.execute(
        RealtimeToolCall("call_1", "add", {"left": 2, "right": 3})
    )
    echoed = await registry.execute(RealtimeToolCall("call_2", "echo", {"ok": True}))

    assert added.output == 5
    assert echoed.output == {"ok": True}
    assert registry.list_names() == ("add", "echo")


@pytest.mark.asyncio
async def test_unknown_tools_are_never_silently_executed() -> None:
    registry = RealtimeToolRegistry()

    with pytest.raises(KeyError, match="No Realtime handler"):
        await execute_realtime_tool_call(
            registry,
            RealtimeToolCall("call_1", "unknown", {}),
            require_approval=False,
        )


@pytest.mark.asyncio
async def test_approval_is_fail_closed_and_hides_arguments_from_repr() -> None:
    call = RealtimeToolCall("call_1", "delete", {"secret": "value"})
    request = RealtimeToolApprovalRequest(
        call_id=call.call_id,
        name=call.name,
        arguments=call.arguments,
        raw=call.raw,
    )

    assert "secret" not in repr(request)
    assert (
        await request_realtime_tool_approval(call, None)
        is RealtimeToolApprovalDecision.REJECT
    )

    async def approve(_: RealtimeToolApprovalRequest) -> RealtimeToolApprovalDecision:
        return RealtimeToolApprovalDecision.APPROVE

    assert (
        await request_realtime_tool_approval(call, approve)
        is RealtimeToolApprovalDecision.APPROVE
    )


@pytest.mark.asyncio
async def test_rejected_tool_can_be_returned_as_explicit_error() -> None:
    registry = RealtimeToolRegistry()
    registry.register("delete", lambda _: "deleted")

    result = await execute_realtime_tool_call(
        registry,
        RealtimeToolCall("call_1", "delete", {}),
        capture_errors=True,
    )

    assert not result.succeeded
    assert isinstance(result.error, PermissionError)


@pytest.mark.asyncio
async def test_tool_timeout_is_explicit_and_cancels_handler() -> None:
    registry = RealtimeToolRegistry()
    cancelled = asyncio.Event()

    async def slow(_: Any) -> str:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unreachable"

    registry.register("slow", slow)

    with pytest.raises(TimeoutError):
        await execute_realtime_tool_call(
            registry,
            RealtimeToolCall("call_1", "slow", {}),
            require_approval=False,
            timeout_seconds=0.01,
        )

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_tool_task_cancellation_propagates() -> None:
    registry = RealtimeToolRegistry()
    started = asyncio.Event()

    async def blocked(_: Any) -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    registry.register("blocked", blocked)
    task = asyncio.create_task(
        execute_realtime_tool_call(
            registry,
            RealtimeToolCall("call_1", "blocked", {}),
            require_approval=False,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_process_tool_event_submits_serialized_output() -> None:
    session = InMemoryRealtimeSession()
    registry = RealtimeToolRegistry()
    registry.register("lookup", lambda arguments: {"query": arguments["query"]})

    async def approve(_: RealtimeToolApprovalRequest) -> RealtimeToolApprovalDecision:
        return RealtimeToolApprovalDecision.APPROVE

    result = await process_realtime_tool_event(
        session,
        registry,
        {
            "call_id": "call_1",
            "name": "lookup",
            "arguments": {"query": "docs"},
        },
        approval_handler=approve,
    )

    assert result.succeeded
    assert session.tool_outputs == [("call_1", '{"query":"docs"}')]


@pytest.mark.asyncio
async def test_explicit_controls_record_only_requested_operations() -> None:
    session = InMemoryRealtimeSession()
    result = RealtimeToolResult("call_1", "lookup", output="done")

    await send_realtime_message(session, " hello ")
    await send_realtime_audio(session, b"audio")
    await interrupt_realtime_session(session)
    await cancel_realtime_response(session)
    await submit_realtime_tool_result(session, result)

    assert session.messages == ["hello"]
    assert session.audio_chunks == [b"audio"]
    assert session.interrupt_calls == 1
    assert session.cancel_calls == 1
    assert session.tool_outputs == [("call_1", "done")]


@pytest.mark.asyncio
async def test_test_session_shutdown_is_idempotent_and_finishes_iteration() -> None:
    session = InMemoryRealtimeSession()
    await session.push_event({"type": "session.created"})
    await session.close()
    await session.close()

    events = [event async for event in session]

    assert events == [{"type": "session.created"}]
    assert session.closed
    assert session.close_calls == 1
    with pytest.raises(RuntimeError, match="finished"):
        await session.push_event({"type": "session.updated"})

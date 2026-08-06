"""Tests for shared operation lifecycle and observability hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from openai_sdk_helpers.agent import runner as agent_runner
from openai_sdk_helpers.codex import CodexPluginContext, CodexPluginRegistry
from openai_sdk_helpers.response import runner as response_runner
from openai_sdk_helpers.runtime import (
    OperationContext,
    OperationEvent,
    OperationPhase,
    run_observed_async,
    run_observed_sync,
)


class EventCollector:
    """Collect lifecycle events for assertions."""

    def __init__(self) -> None:
        self.events: list[OperationEvent] = []

    def __call__(self, event: OperationEvent) -> None:
        self.events.append(event)


@dataclass
class FakeUsage:
    """SDK-shaped usage fixture."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class FakeResult:
    """SDK-shaped result fixture."""

    model: str = "example-model"
    request_id: str = "request-123"
    usage: FakeUsage = FakeUsage(3, 5, 8)


def test_sync_lifecycle_captures_usage_and_safe_diagnostics() -> None:
    collector = EventCollector()
    context = OperationContext(
        "responses.run_sync",
        correlation_id="correlation-1",
        trace_id="trace-1",
        metadata={
            "tenant": "example",
            "api_key": "not-a-real-secret",
            "prompt": "sensitive text",
        },
        observers=(collector,),
    )
    result = FakeResult()

    returned = run_observed_sync(context, lambda: result)

    assert returned is result
    assert [event.phase for event in collector.events] == [
        OperationPhase.START,
        OperationPhase.SUCCESS,
    ]
    success = collector.events[-1]
    assert success.result is result
    assert success.model == "example-model"
    assert success.request_id == "request-123"
    assert success.usage is not None
    assert success.usage.total_tokens == 8
    diagnostics = success.diagnostics()
    assert diagnostics["metadata"] == {
        "tenant": "example",
        "api_key": "<redacted>",
        "prompt": "<redacted>",
    }
    assert "result" not in diagnostics


def test_failure_preserves_original_exception() -> None:
    collector = EventCollector()
    context = OperationContext("codex.failure", observers=(collector,))
    error = ValueError("sensitive failure detail")

    def fail() -> None:
        raise error

    with pytest.raises(ValueError) as exc_info:
        run_observed_sync(context, fail)

    assert exc_info.value is error
    assert collector.events[-1].error is error
    diagnostics = collector.events[-1].diagnostics()
    assert diagnostics["error_type"] == "ValueError"
    assert "sensitive failure detail" not in str(diagnostics)


@pytest.mark.asyncio
async def test_async_lifecycle_matches_sync_phases() -> None:
    collector = EventCollector()
    context = OperationContext("agents.run_async", observers=(collector,))

    async def execute() -> str:
        return "done"

    result = await run_observed_async(context, execute)

    assert result == "done"
    assert [event.phase for event in collector.events] == [
        OperationPhase.START,
        OperationPhase.SUCCESS,
    ]


def test_observer_failure_does_not_replace_operation_result() -> None:
    def broken_observer(_: OperationEvent) -> None:
        raise RuntimeError("observer unavailable")

    context = OperationContext("safe.observer", observers=(broken_observer,))

    result = run_observed_sync(context, lambda: "ok")

    assert result == "ok"
    assert context.observer_error_count == 2


def test_response_runner_observes_execution_and_cleanup() -> None:
    collector = EventCollector()
    context = OperationContext("responses.runner", observers=(collector,))

    class FakeResponse:
        closed = False

        def __init__(self, **_: Any) -> None:
            pass

        def run_sync(self, *, content: str) -> dict[str, object]:
            return {"content": content, "usage": {"total_tokens": 2}}

        def close(self) -> None:
            type(self).closed = True

    result = response_runner.run_sync(
        FakeResponse,
        content="hello",
        operation_context=context,
    )

    assert result["content"] == "hello"
    assert FakeResponse.closed is True
    assert collector.events[-1].usage is not None
    assert collector.events[-1].usage.total_tokens == 2


@pytest.mark.asyncio
async def test_agent_runner_emits_events_without_changing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = EventCollector()
    context = OperationContext("agents.runner", observers=(collector,))
    expected = FakeResult()

    async def fake_run(*_: Any, **__: Any) -> FakeResult:
        return expected

    monkeypatch.setattr(agent_runner.Runner, "run", fake_run)

    result = await agent_runner.run_async(
        object(),
        "hello",
        operation_context=context,
    )

    assert result is expected
    assert collector.events[-1].result is expected


@pytest.mark.asyncio
async def test_codex_registry_observes_sync_and_async_commands() -> None:
    class Plugin:
        name = "example"

        def setup(self, context: CodexPluginContext) -> None:
            context.add_command("echo", lambda value: value)

            async def echo_async(value: str) -> str:
                return value

            context.add_command("echo-async", echo_async)

    registry = CodexPluginRegistry()
    registry.register(Plugin())

    sync_collector = EventCollector()
    sync_context = OperationContext("codex.echo", observers=(sync_collector,))
    assert registry.run("echo", "sync", operation_context=sync_context) == "sync"
    assert sync_collector.events[-1].phase is OperationPhase.SUCCESS

    async_collector = EventCollector()
    async_context = OperationContext(
        "codex.echo-async",
        observers=(async_collector,),
    )
    result = await registry.run_async(
        "echo-async",
        "async",
        operation_context=async_context,
    )
    assert result == "async"
    assert async_collector.events[-1].phase is OperationPhase.SUCCESS

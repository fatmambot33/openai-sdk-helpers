"""Tests for explicit server-side Realtime session lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openai_sdk_helpers import OperationContext, OperationEvent, OperationPhase
from openai_sdk_helpers.realtime import (
    ManagedRealtimeSession,
    RealtimeLifecycleConfig,
    RealtimeLifecycleState,
    RealtimeRunnerConfig,
    RealtimeSessionConfig,
    build_realtime_runner,
    manage_realtime_runner,
)
from openai_sdk_helpers.realtime import session as session_module


class FakeSession:
    """SDK-shaped Realtime session fixture."""

    def __init__(self) -> None:
        self.close_calls = 0
        self.close_error: BaseException | None = None
        self.close_gate: asyncio.Event | None = None

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_gate is not None:
            await self.close_gate.wait()
        if self.close_error is not None:
            raise self.close_error


class FakeRunner:
    """SDK-shaped Realtime runner fixture."""

    def __init__(self) -> None:
        self.run_calls = 0
        self.sessions: list[FakeSession] = []
        self.run_error: BaseException | None = None
        self.run_gate: asyncio.Event | None = None

    async def run(self) -> FakeSession:
        self.run_calls += 1
        if self.run_gate is not None:
            await self.run_gate.wait()
        if self.run_error is not None:
            raise self.run_error
        session = FakeSession()
        self.sessions.append(session)
        return session


class Collector:
    """Collect operation lifecycle events."""

    def __init__(self) -> None:
        self.events: list[OperationEvent] = []

    def __call__(self, event: OperationEvent) -> None:
        self.events.append(event)


def test_session_config_serializes_only_explicit_values() -> None:
    config = RealtimeSessionConfig(
        model="realtime-model",
        voice="voice-name",
        modalities=("audio", "text", "audio"),
        turn_detection={"type": "server_vad"},
        tool_choice={"type": "function", "name": "lookup"},
        extra={"max_output_tokens": 256},
    )

    assert config.as_model_config() == {
        "max_output_tokens": 256,
        "model": "realtime-model",
        "voice": "voice-name",
        "modalities": ["audio", "text"],
        "turn_detection": {"type": "server_vad"},
        "tool_choice": {"type": "function", "name": "lookup"},
    }


def test_runner_config_preserves_official_passthrough() -> None:
    config = RealtimeRunnerConfig(
        session=RealtimeSessionConfig(instructions="Answer clearly."),
        workflow_name="support",
        group_id="conversation-42",
        trace_metadata={"tenant": "example"},
        tracing_disabled=True,
        extra={"custom_setting": True},
    )

    assert config.as_sdk_config() == {
        "custom_setting": True,
        "model_settings": {"instructions": "Answer clearly."},
        "workflow_name": "support",
        "group_id": "conversation-42",
        "trace_metadata": {"tenant": "example"},
        "tracing_disabled": True,
    }


def test_builder_returns_raw_official_runner_without_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeOfficialRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        session_module,
        "_load_realtime_runner_type",
        lambda: FakeOfficialRunner,
    )
    agent = object()

    runner = build_realtime_runner(
        agent,
        config=RealtimeRunnerConfig(
            session=RealtimeSessionConfig(model="realtime-model")
        ),
    )

    assert isinstance(runner, FakeOfficialRunner)
    assert captured == {
        "starting_agent": agent,
        "config": {"model_settings": {"model": "realtime-model"}},
    }


@pytest.mark.asyncio
async def test_lifecycle_is_explicit_idempotent_and_preserves_raw_objects() -> None:
    runner = FakeRunner()
    managed = manage_realtime_runner(runner)

    assert managed.raw_runner is runner
    assert managed.raw_session is None
    assert managed.state is RealtimeLifecycleState.CREATED

    session = await managed.start()
    assert session is runner.sessions[0]
    assert managed.raw_session is session
    assert managed.active is True
    assert await managed.start() is session
    assert runner.run_calls == 1

    await managed.close()
    await managed.close()
    assert session.close_calls == 1
    assert managed.state is RealtimeLifecycleState.CLOSED


@pytest.mark.asyncio
async def test_async_context_returns_raw_session_and_closes() -> None:
    runner = FakeRunner()
    managed = ManagedRealtimeSession(runner)

    async with managed as session:
        assert session is runner.sessions[0]
        assert managed.active is True

    assert session.close_calls == 1
    assert managed.state is RealtimeLifecycleState.CLOSED


@pytest.mark.asyncio
async def test_restart_is_disabled_by_default_and_explicit_when_enabled() -> None:
    runner = FakeRunner()
    managed = ManagedRealtimeSession(runner)

    await managed.start()
    await managed.close()
    with pytest.raises(RuntimeError, match="restart is disabled"):
        await managed.start()

    restartable = ManagedRealtimeSession(
        runner,
        lifecycle=RealtimeLifecycleConfig(allow_restart=True),
    )
    first = await restartable.start()
    second = await restartable.restart()

    assert first is not second
    assert first.close_calls == 1
    assert restartable.raw_session is second
    assert restartable.active is True


@pytest.mark.asyncio
async def test_start_failure_preserves_exception_and_observability() -> None:
    runner = FakeRunner()
    error = ConnectionError("realtime unavailable")
    runner.run_error = error
    collector = Collector()
    context = OperationContext("realtime.start", observers=(collector,))
    managed = ManagedRealtimeSession(runner)

    with pytest.raises(ConnectionError) as exc_info:
        await managed.start(operation_context=context)

    assert exc_info.value is error
    assert managed.state is RealtimeLifecycleState.FAILED
    assert [event.phase for event in collector.events] == [
        OperationPhase.START,
        OperationPhase.FAILURE,
    ]
    assert collector.events[-1].error is error


@pytest.mark.asyncio
async def test_start_timeout_is_explicit_and_marks_failure() -> None:
    runner = FakeRunner()
    runner.run_gate = asyncio.Event()
    managed = ManagedRealtimeSession(
        runner,
        lifecycle=RealtimeLifecycleConfig(start_timeout_seconds=0.01),
    )

    with pytest.raises(asyncio.TimeoutError):
        await managed.start()

    assert managed.state is RealtimeLifecycleState.FAILED
    assert managed.raw_session is None


@pytest.mark.asyncio
async def test_close_failure_and_timeout_preserve_state() -> None:
    runner = FakeRunner()
    managed = ManagedRealtimeSession(runner)
    session = await managed.start()
    error = RuntimeError("close failed")
    session.close_error = error

    with pytest.raises(RuntimeError) as exc_info:
        await managed.close()
    assert exc_info.value is error
    assert managed.state is RealtimeLifecycleState.FAILED

    timeout_runner = FakeRunner()
    timeout_managed = ManagedRealtimeSession(
        timeout_runner,
        lifecycle=RealtimeLifecycleConfig(close_timeout_seconds=0.01),
    )
    timeout_session = await timeout_managed.start()
    timeout_session.close_gate = asyncio.Event()

    with pytest.raises(asyncio.TimeoutError):
        await timeout_managed.close()
    assert timeout_managed.state is RealtimeLifecycleState.FAILED


def test_realtime_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        RealtimeSessionConfig(model=" ")
    with pytest.raises(ValueError, match="modality must not be empty"):
        RealtimeSessionConfig(modalities=("audio", " "))
    with pytest.raises(ValueError, match="positive"):
        RealtimeLifecycleConfig(start_timeout_seconds=0)
    with pytest.raises(ValueError, match="positive"):
        RealtimeLifecycleConfig(close_timeout_seconds=-1)

"""Tests for the CoordinatorAgent class."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from openai_sdk_helpers.structure.plan.enum import AgentEnum
from openai_sdk_helpers.agent.coordinator import CoordinatorAgent
from openai_sdk_helpers.structure import (
    TaskStructure,
    PlanStructure,
    PromptStructure,
)


@pytest.fixture
def mock_prompt_fn():
    """Return a mock prompt_fn."""
    return MagicMock(return_value=PromptStructure(prompt="test brief"))


@pytest.fixture
def mock_build_plan_fn():
    """Return a mock build_plan_fn."""
    return MagicMock(return_value=PlanStructure())


@pytest.fixture
def mock_execute_plan_fn():
    """Return a mock execute_plan_fn."""
    return MagicMock(return_value=["test result"])


@pytest.fixture
def mock_summarize_fn():
    """Return a mock summarize_fn."""
    return MagicMock(return_value="test summary")


@pytest.fixture
def coordinator_agent(
    tmp_path,
    mock_prompt_fn,
    mock_build_plan_fn,
    mock_execute_plan_fn,
    mock_summarize_fn,
):
    """Return an CoordinatorAgent instance."""
    with patch("openai_sdk_helpers.agent.coordinator.CoordinatorAgent.save"):
        yield CoordinatorAgent(
            prompt_fn=mock_prompt_fn,
            build_plan_fn=mock_build_plan_fn,
            execute_plan_fn=mock_execute_plan_fn,
            summarize_fn=mock_summarize_fn,
            module_data_path=tmp_path,
            name="test_module",
            model="test_model",
        )


def test_coordinator_agent_initialization(coordinator_agent):
    """Test ProjectManager initialization."""
    assert coordinator_agent.prompt is None
    assert coordinator_agent.brief is None
    assert coordinator_agent.plan == PlanStructure()
    assert coordinator_agent.summary is None


def test_build_prompt(coordinator_agent, mock_prompt_fn):
    """Test building instructions."""
    coordinator_agent.build_prompt("test prompt")
    assert coordinator_agent.prompt == "test prompt"
    mock_prompt_fn.assert_called_once_with("test prompt")
    assert coordinator_agent.brief == PromptStructure(prompt="test brief")


def test_build_plan(coordinator_agent, mock_build_plan_fn):
    """Test building a plan."""
    coordinator_agent.brief = PromptStructure(prompt="test brief")
    coordinator_agent.build_plan()
    mock_build_plan_fn.assert_called_once_with("test brief")
    assert coordinator_agent.plan == PlanStructure()


def test_build_plan_no_brief(coordinator_agent):
    """Test that building a plan without a brief raises an error."""
    with pytest.raises(ValueError):
        coordinator_agent.build_plan()


def test_execute_plan(coordinator_agent, mock_execute_plan_fn):
    """Test executing a plan."""
    task = TaskStructure(prompt="test task")
    coordinator_agent.plan = PlanStructure(tasks=[task])
    coordinator_agent.execute_plan()
    mock_execute_plan_fn.assert_called_once_with(coordinator_agent.plan)


def test_summarize_plan(coordinator_agent, mock_summarize_fn):
    """Test summarizing a plan."""
    summary = coordinator_agent.summarize_plan(["test result"])
    mock_summarize_fn.assert_called_once_with(["test result"])
    assert summary == "test summary"


def test_summarize_plan_no_results(coordinator_agent):
    """Test summarizing a plan with no results."""
    summary = coordinator_agent.summarize_plan()
    assert summary == ""


def test_run_plan(coordinator_agent):
    """Test running a full plan."""
    coordinator_agent.build_prompt = MagicMock()
    coordinator_agent.build_plan = MagicMock()
    coordinator_agent.execute_plan = MagicMock()
    coordinator_agent.summarize_plan = MagicMock()
    coordinator_agent.run_plan("test prompt")
    coordinator_agent.build_prompt.assert_called_once_with("test prompt")
    coordinator_agent.build_plan.assert_called_once()
    coordinator_agent.execute_plan.assert_called_once()
    coordinator_agent.summarize_plan.assert_called_once()


def test_run_task(coordinator_agent):
    """Test running a single task."""
    task = TaskStructure(prompt="test task", task_type=AgentEnum.WEB_SEARCH)

    def agent_callable(*args, **kwargs):
        return "test output"

    result = coordinator_agent._run_task(task, agent_callable, [])
    assert result == "test output"


def test_run_task_in_thread(coordinator_agent):
    """Test running a task in a thread."""
    task = TaskStructure(prompt="test task", task_type=AgentEnum.WEB_SEARCH)

    def agent_callable(*args, **kwargs):
        return "test output"

    result = coordinator_agent._run_task_in_thread(task, agent_callable, [])
    assert result == "test output"


def test_resolve_result(coordinator_agent):
    """Test resolving a result."""
    result = coordinator_agent._resolve_result("test result")
    assert result == "test result"


def test_normalize_results(coordinator_agent):
    """Test normalizing results."""
    assert coordinator_agent._normalize_results(None) == []
    assert coordinator_agent._normalize_results("test") == ["test"]
    assert coordinator_agent._normalize_results(["test1", "test2"]) == [
        "test1",
        "test2",
    ]


def test_resolve_result_handles_completed_future(coordinator_agent):
    """ProjectManager should unwrap results from completed futures."""
    loop = asyncio.new_event_loop()
    future: asyncio.Future[str] = loop.create_future()
    future.set_result("ready")

    assert coordinator_agent._resolve_result(future) == "ready"
    loop.close()


def test_resolve_result_waits_on_running_loop_future(coordinator_agent):
    """ProjectManager should wait for futures tied to running event loops."""

    loop = asyncio.new_event_loop()
    result: asyncio.Future[str] = loop.create_future()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    loop.call_soon_threadsafe(result.set_result, "from-loop")

    try:
        assert coordinator_agent._resolve_result(result) == "from-loop"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def test_resolve_result_rejects_pending_future_on_own_loop(coordinator_agent):
    """_resolve_result should avoid blocking the loop that owns a pending task."""

    async def runner() -> None:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(future.set_result, "loop-value")

        with pytest.raises(RuntimeError, match="owning running event loop"):
            coordinator_agent._resolve_result(future)

        # Ensure the loop processes the scheduled result to avoid warnings.
        assert await future == "loop-value"

    asyncio.run(runner())


def test_run_task_in_thread_awaits_async_callable(coordinator_agent):
    """_run_task_in_thread should await asynchronous agent callables."""

    async def async_agent(prompt: str, context: list[str] | None = None) -> str:
        await asyncio.sleep(0)
        return f"{prompt}::{len(context or [])}"

    task = TaskStructure(prompt="async-task", task_type=AgentEnum.SUMMARIZER)
    result = coordinator_agent._run_task_in_thread(task, async_agent, ["ctx"])

    assert result == "async-task\n\nContext:\nctx::1"

"""Tests for the ProjectManager class."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from openai_sdk_helpers.structure.plan.enum import AgentEnum
from openai_sdk_helpers.agent.project_manager import ProjectManager
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
def project_manager(
    tmp_path,
    mock_prompt_fn,
    mock_build_plan_fn,
    mock_execute_plan_fn,
    mock_summarize_fn,
):
    """Return a ProjectManager instance."""
    with patch("openai_sdk_helpers.agent.project_manager.ProjectManager.save"):
        yield ProjectManager(
            prompt_fn=mock_prompt_fn,
            build_plan_fn=mock_build_plan_fn,
            execute_plan_fn=mock_execute_plan_fn,
            summarize_fn=mock_summarize_fn,
            module_data_path=tmp_path,
            module_name="test_module",
            default_model="test_model",
        )


def test_project_manager_initialization(project_manager):
    """Test ProjectManager initialization."""
    assert project_manager.prompt is None
    assert project_manager.brief is None
    assert project_manager.plan == PlanStructure()
    assert project_manager.summary is None


def test_build_prompt(project_manager, mock_prompt_fn):
    """Test building instructions."""
    project_manager.build_prompt("test prompt")
    assert project_manager.prompt == "test prompt"
    mock_prompt_fn.assert_called_once_with("test prompt")
    assert project_manager.brief == PromptStructure(prompt="test brief")


def test_build_plan(project_manager, mock_build_plan_fn):
    """Test building a plan."""
    project_manager.brief = PromptStructure(prompt="test brief")
    project_manager.build_plan()
    mock_build_plan_fn.assert_called_once_with("test brief")
    assert project_manager.plan == PlanStructure()


def test_build_plan_no_brief(project_manager):
    """Test that building a plan without a brief raises an error."""
    with pytest.raises(ValueError):
        project_manager.build_plan()


def test_execute_plan(project_manager, mock_execute_plan_fn):
    """Test executing a plan."""
    task = TaskStructure(prompt="test task")
    project_manager.plan = PlanStructure(tasks=[task])
    project_manager.execute_plan()
    mock_execute_plan_fn.assert_called_once_with(project_manager.plan)


def test_summarize_plan(project_manager, mock_summarize_fn):
    """Test summarizing a plan."""
    summary = project_manager.summarize_plan(["test result"])
    mock_summarize_fn.assert_called_once_with(["test result"])
    assert summary == "test summary"


def test_summarize_plan_no_results(project_manager):
    """Test summarizing a plan with no results."""
    summary = project_manager.summarize_plan()
    assert summary == ""


def test_run_plan(project_manager):
    """Test running a full plan."""
    project_manager.build_prompt = MagicMock()
    project_manager.build_plan = MagicMock()
    project_manager.execute_plan = MagicMock()
    project_manager.summarize_plan = MagicMock()
    project_manager.run_plan("test prompt")
    project_manager.build_prompt.assert_called_once_with("test prompt")
    project_manager.build_plan.assert_called_once()
    project_manager.execute_plan.assert_called_once()
    project_manager.summarize_plan.assert_called_once()


def test_file_path(project_manager):
    """Test the file_path property."""
    assert project_manager.file_path.name.endswith(".json")


def test_run_task(project_manager):
    """Test running a single task."""
    task = TaskStructure(prompt="test task", task_type=AgentEnum.WEB_SEARCH)
    agent_callable = MagicMock(return_value="test output")
    result = project_manager._run_task(task, agent_callable, [])
    assert result == "test output"


def test_run_task_in_thread(project_manager):
    """Test running a task in a thread."""
    task = TaskStructure(prompt="test task", task_type=AgentEnum.WEB_SEARCH)
    agent_callable = MagicMock(return_value="test output")
    result = project_manager._run_task_in_thread(task, agent_callable, [])
    assert result == "test output"


def test_resolve_result(project_manager):
    """Test resolving a result."""
    result = project_manager._resolve_result("test result")
    assert result == "test result"


def test_normalize_results(project_manager):
    """Test normalizing results."""
    assert project_manager._normalize_results(None) == []
    assert project_manager._normalize_results("test") == ["test"]
    assert project_manager._normalize_results(["test1", "test2"]) == ["test1", "test2"]


def test_resolve_result_handles_completed_future(project_manager):
    """ProjectManager should unwrap results from completed futures."""
    loop = asyncio.new_event_loop()
    future: asyncio.Future[str] = loop.create_future()
    future.set_result("ready")

    assert project_manager._resolve_result(future) == "ready"
    loop.close()


def test_resolve_result_waits_on_running_loop_future(project_manager):
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
        assert project_manager._resolve_result(result) == "from-loop"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def test_run_task_in_thread_awaits_async_callable(project_manager):
    """_run_task_in_thread should await asynchronous agent callables."""

    async def async_agent(prompt: str, context: list[str] | None = None) -> str:
        await asyncio.sleep(0)
        return f"{prompt}::{len(context or [])}"

    task = TaskStructure(prompt="async-task", task_type=AgentEnum.SUMMARIZER)
    result = project_manager._run_task_in_thread(task, async_agent, ["ctx"])

    assert result == "async-task\n\nContext:\nctx::1"

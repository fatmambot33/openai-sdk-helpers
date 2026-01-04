"""Tests for plan helper functions."""

from __future__ import annotations

import pytest

from openai_sdk_helpers.structure.plan import (
    AgentEnum,
    PlanStructure,
    TaskStructure,
    create_plan,
    execute_task,
    execute_plan,
)


def test_create_plan_basic():
    """Test create_plan factory function."""
    task1 = TaskStructure(
        task_type=AgentEnum.WEB_SEARCH, prompt="Search for something"
    )
    task2 = TaskStructure(
        task_type=AgentEnum.SUMMARIZER, prompt="Summarize results"
    )

    plan = create_plan(task1, task2)

    assert isinstance(plan, PlanStructure)
    assert len(plan) == 2
    assert plan.tasks[0].prompt == "Search for something"
    assert plan.tasks[1].prompt == "Summarize results"


def test_create_plan_empty():
    """Test create_plan with no tasks."""
    plan = create_plan()

    assert isinstance(plan, PlanStructure)
    assert len(plan) == 0


def test_create_plan_single_task():
    """Test create_plan with one task."""
    task = TaskStructure(task_type=AgentEnum.WEB_SEARCH, prompt="Single task")

    plan = create_plan(task)

    assert len(plan) == 1
    assert plan.tasks[0].prompt == "Single task"


def test_execute_task_basic():
    """Test execute_task with a simple agent callable."""

    def mock_agent(prompt, context=None):
        return f"Result for: {prompt}"

    task = TaskStructure(task_type=AgentEnum.WEB_SEARCH, prompt="Test prompt")

    results = execute_task(task, mock_agent)

    assert isinstance(results, list)
    assert len(results) > 0
    assert task.status == "done"
    assert task.start_date is not None
    assert task.end_date is not None


def test_execute_task_with_context():
    """Test execute_task passes context correctly."""

    def mock_agent(prompt, context=None):
        if context:
            return f"Prompt: {prompt}, Context: {','.join(context)}"
        return f"Prompt: {prompt}"

    task = TaskStructure(task_type=AgentEnum.SUMMARIZER, prompt="Summarize")

    results = execute_task(task, mock_agent, context=["prev result 1", "prev result 2"])

    # Results should include context information
    assert len(results) > 0


def test_execute_task_error_handling():
    """Test that execute_task properly handles errors."""

    def failing_agent(prompt, context=None):
        raise ValueError("Agent failed")

    task = TaskStructure(task_type=AgentEnum.WEB_SEARCH, prompt="Will fail")

    # The error should be raised after being recorded in task
    with pytest.raises(ValueError, match="Agent failed"):
        execute_task(task, failing_agent)

    # Task should be marked as error even though exception was raised
    assert task.status == "error"
    assert len(task.results) > 0


def test_execute_plan_wrapper():
    """Test execute_plan convenience wrapper."""

    def mock_search(prompt, context=None):
        return ["search result"]

    def mock_summarize(prompt, context=None):
        return ["summary result"]

    registry = {
        AgentEnum.WEB_SEARCH: mock_search,
        AgentEnum.SUMMARIZER: mock_summarize,
    }

    plan = PlanStructure(
        tasks=[
            TaskStructure(task_type=AgentEnum.WEB_SEARCH, prompt="Search"),
            TaskStructure(task_type=AgentEnum.SUMMARIZER, prompt="Summarize"),
        ]
    )

    results = execute_plan(plan, registry)

    assert isinstance(results, list)
    assert len(results) == 2
    assert "search result" in results
    assert "summary result" in results


def test_execute_plan_halt_on_error():
    """Test that execute_plan respects halt_on_error parameter."""

    def failing_agent(prompt, context=None):
        raise ValueError("Task failed")

    def success_agent(prompt, context=None):
        return ["success"]

    registry = {
        AgentEnum.WEB_SEARCH: failing_agent,
        AgentEnum.SUMMARIZER: success_agent,
    }

    plan = PlanStructure(
        tasks=[
            TaskStructure(task_type=AgentEnum.WEB_SEARCH, prompt="Will fail"),
            TaskStructure(task_type=AgentEnum.SUMMARIZER, prompt="Should not run"),
        ]
    )

    # With halt_on_error=True (default), should stop after first error
    results = execute_plan(plan, registry, halt_on_error=True)

    # First task should have error status
    assert plan.tasks[0].status == "error"
    # Second task should still be waiting (not executed)
    assert plan.tasks[1].status == "waiting"


def test_execute_plan_continue_on_error():
    """Test execute_plan continues when halt_on_error=False."""

    def failing_agent(prompt, context=None):
        raise ValueError("Task failed")

    def success_agent(prompt, context=None):
        return ["success"]

    registry = {
        AgentEnum.WEB_SEARCH: failing_agent,
        AgentEnum.SUMMARIZER: success_agent,
    }

    plan = PlanStructure(
        tasks=[
            TaskStructure(task_type=AgentEnum.WEB_SEARCH, prompt="Will fail"),
            TaskStructure(task_type=AgentEnum.SUMMARIZER, prompt="Should run"),
        ]
    )

    # With halt_on_error=False, should continue after error
    results = execute_plan(plan, registry, halt_on_error=False)

    # Both tasks should have completed (one error, one success)
    assert plan.tasks[0].status == "error"
    assert plan.tasks[1].status == "done"
    assert "success" in results


def test_execute_plan_missing_agent():
    """Test that execute_plan raises KeyError for missing agent."""
    registry = {}  # Empty registry

    plan = PlanStructure(
        tasks=[TaskStructure(task_type=AgentEnum.WEB_SEARCH, prompt="Test")]
    )

    with pytest.raises(KeyError, match="No agent registered"):
        execute_plan(plan, registry)

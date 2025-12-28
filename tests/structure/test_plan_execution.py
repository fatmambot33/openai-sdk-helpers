"""Tests for executing plan structures with registered agents."""

from __future__ import annotations

import pytest

from openai_sdk_helpers.structure import AgentEnum, AgentTaskStructure, PlanStructure


def test_execute_runs_tasks_and_tracks_status():
    """Run tasks sequentially and capture status transitions and context."""

    captured_context = []

    def designer_agent(prompt: str, context: list[str] | None = None) -> str:
        captured_context.append(("designer", prompt, list(context or [])))
        return "design draft"

    def builder_agent(prompt: str, context: list[str] | None = None) -> list[str]:
        captured_context.append(("builder", prompt, list(context or [])))
        return ["artifact", "tests"]

    plan = PlanStructure(
        tasks=[
            AgentTaskStructure(task_type=AgentEnum.DESIGNER, prompt="Design agent"),
            AgentTaskStructure(task_type=AgentEnum.BUILDER, prompt="Build agent"),
        ]
    )

    results = plan.execute(
        {
            AgentEnum.DESIGNER: designer_agent,
            AgentEnum.BUILDER: builder_agent,
        }
    )

    assert results == ["design draft", "artifact", "tests"]

    first_task, second_task = plan.tasks
    assert first_task.status == "done"
    assert second_task.status == "done"
    assert first_task.start_date is not None and first_task.end_date is not None
    assert second_task.start_date is not None and second_task.end_date is not None

    assert captured_context[0][2] == []
    assert "design draft" in captured_context[1][2]


def test_execute_raises_when_agent_missing():
    """Raise a KeyError when a task lacks a registered agent callable."""

    plan = PlanStructure(
        tasks=[
            AgentTaskStructure(
                task_type=AgentEnum.DESIGNER,
                prompt="Design agent",
            )
        ]
    )

    with pytest.raises(KeyError):
        plan.execute({})

    assert plan.tasks[0].status == "waiting"
    assert plan.tasks[0].start_date is None


def test_execute_continues_on_error_when_configured():
    """Continue executing tasks when ``halt_on_error`` is False."""

    def failing_agent(
        prompt: str, context: list[str] | None = None
    ) -> str:  # noqa: ARG001
        raise ValueError("boom")

    def builder_agent(prompt: str, context: list[str] | None = None) -> str:
        return f"built from {context[-1]}" if context else "built"

    plan = PlanStructure(
        tasks=[
            AgentTaskStructure(task_type=AgentEnum.DESIGNER, prompt="Design agent"),
            AgentTaskStructure(task_type=AgentEnum.BUILDER, prompt="Build agent"),
        ]
    )

    results = plan.execute(
        {
            AgentEnum.DESIGNER: failing_agent,
            AgentEnum.BUILDER: builder_agent,
        },
        halt_on_error=False,
    )

    assert results[0].startswith("Task error: boom")
    assert results[1] == f"built from {results[0]}"

    first_task, second_task = plan.tasks
    assert first_task.status == "error"
    assert second_task.status == "done"
    assert first_task.start_date is not None and first_task.end_date is not None
    assert second_task.start_date is not None and second_task.end_date is not None


def test_execute_forwards_task_context_alongside_history():
    """Forward static task context and accumulated results to agents."""

    captured = []

    def planner_agent(prompt: str, context: list[str] | None = None) -> str:
        captured.append(("planner", prompt, list(context or [])))
        return "scoped"

    def validator_agent(prompt: str, context: list[str] | None = None) -> str:
        captured.append(("validator", prompt, list(context or [])))
        return "validated"

    plan = PlanStructure(
        tasks=[
            AgentTaskStructure(
                task_type=AgentEnum.PLANNER,
                prompt="Scope mission",
                context=["guardrail: limit scope"],
            ),
            AgentTaskStructure(
                task_type=AgentEnum.VALIDATOR,
                prompt="Review outputs",
                context=["tool: safety-check"],
            ),
        ]
    )

    results = plan.execute(
        {
            AgentEnum.PLANNER: planner_agent,
            AgentEnum.VALIDATOR: validator_agent,
        }
    )

    assert results == ["scoped", "validated"]

    planner_call, validator_call = captured
    assert planner_call[2] == ["guardrail: limit scope"]
    assert planner_call[1].endswith("guardrail: limit scope")

    assert validator_call[2] == ["tool: safety-check", "scoped"]
    assert "tool: safety-check" in validator_call[1]

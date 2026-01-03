"""Integration tests for the openai-sdk-helpers package."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from openai_sdk_helpers.agent.coordination import CoordinatorAgent
from openai_sdk_helpers.structure import (
    TaskStructure,
    PlanStructure,
    PromptStructure,
)


def test_project_manager_integration(tmp_path):
    """Test the CoordinatorAgent integration."""

    def prompt_fn(prompt: str) -> PromptStructure:
        return PromptStructure(prompt=f"Brief for {prompt}")

    def build_plan_fn(brief: str) -> PlanStructure:
        return PlanStructure(
            tasks=[
                TaskStructure(prompt=f"Task 1 for {brief}"),
                TaskStructure(prompt=f"Task 2 for {brief}"),
            ]
        )

    def execute_plan_fn(plan: PlanStructure) -> list[str]:
        results = []
        for task in plan.tasks:
            results.append(f"Result for {task.prompt}")
        return results

    def summarize_fn(results: list[str]) -> str:
        return f"Summary of {', '.join(results)}"

    with patch("openai_sdk_helpers.agent.project_manager.CoordinatorAgent.save"):
        pm = CoordinatorAgent(
            prompt_fn=prompt_fn,
            build_plan_fn=build_plan_fn,
            execute_plan_fn=execute_plan_fn,
            summarize_fn=summarize_fn,
            module_data_path=tmp_path,
            name="test_module",
            default_model="test_model",
        )

        pm.run_plan("test prompt")

        assert pm.prompt == "test prompt"
        assert pm.brief == PromptStructure(prompt="Brief for test prompt")
        assert len(pm.plan.tasks) == 2
        assert (
            pm.summary
            == "Summary of Result for Task 1 for Brief for test prompt, Result for Task 2 for Brief for test prompt"
        )

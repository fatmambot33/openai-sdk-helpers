"""Integration tests for the corai package."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from corai.agent.project_manager import ProjectManager
from corai.structure import AgentTaskStructure, PlanStructure, PromptStructure


def test_project_manager_integration(tmp_path):
    """Test the ProjectManager integration."""

    def build_brief_fn(prompt: str) -> PromptStructure:
        return PromptStructure(prompt=f"Brief for {prompt}")

    def build_plan_fn(brief: str) -> PlanStructure:
        return PlanStructure(
            tasks=[
                AgentTaskStructure(prompt=f"Task 1 for {brief}"),
                AgentTaskStructure(prompt=f"Task 2 for {brief}"),
            ]
        )

    def execute_plan_fn(plan: PlanStructure) -> list[str]:
        results = []
        for task in plan.tasks:
            results.append(f"Result for {task.prompt}")
        return results

    def summarize_fn(results: list[str]) -> str:
        return f"Summary of {', '.join(results)}"

    with patch("corai.agent.project_manager.ProjectManager.save"):
        pm = ProjectManager(
            build_brief_fn=build_brief_fn,
            build_plan_fn=build_plan_fn,
            execute_plan_fn=execute_plan_fn,
            summarize_fn=summarize_fn,
            module_data_path=tmp_path,
            module_name="test_module",
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

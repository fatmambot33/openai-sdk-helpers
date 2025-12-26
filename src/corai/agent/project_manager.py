"""Generic project manager for coordinating agent plans."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..structure import AgentEnum, AgentTaskStructure, PlanStructure, PromptStructure
from ..environment import DATETIME_FMT
from ..utils import JSONSerializable, log
from .base import BaseAgent
from .config import AgentConfig

BuildBriefFn = Callable[[str], PromptStructure]
BuildPlanFn = Callable[[str], PlanStructure]
ExecutePlanFn = Callable[[PlanStructure], List[str]]
SummarizeFn = Callable[[List[str]], str]


class ProjectManager(BaseAgent, JSONSerializable):
    """Coordinate agent plans while persisting project state and outputs.

    Methods
    -------
    build_instructions(prompt)
        Summarize the prompt into a concise brief.
    build_plan()
        Create a list of ``AgentTaskStructure`` entries for the project.
    execute_plan()
        Run each task sequentially while tracking status and timing.
    summarize_plan(results)
        Summarize a collection of result strings.
    to_dict()
        Return a JSON-serializable snapshot of stored project data.
    save()
        Persist the stored project data to a JSON file.
    """

    def __init__(
        self,
        build_brief_fn: BuildBriefFn,
        build_plan_fn: BuildPlanFn,
        execute_plan_fn: ExecutePlanFn,
        summarize_fn: SummarizeFn,
        module_data_path: Path,
        module_name: str,
        config: Optional[AgentConfig] = None,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the project manager with injected workflow helpers.

        Parameters
        ----------
        build_brief_fn
            Callable that generates a prompt brief from the input string.
        build_plan_fn
            Callable that generates a plan from the prompt brief.
        execute_plan_fn
            Callable that executes a plan and returns results.
        summarize_fn
            Callable that summarizes a list of result strings.
        module_data_path
            Base path for persisting project artifacts.
        module_name
            Name of the parent module for data organization.
        config
            Optional agent configuration describing prompts and metadata.
        prompt_dir
            Optional directory holding prompt templates.
        default_model
            Optional fallback model identifier.

        Returns
        -------
        None
        """
        if config is None:
            config = AgentConfig(
                name="project_manager",
                description="Coordinates agents for planning and summarization.",
            )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )
        self._build_brief_fn = build_brief_fn
        self._build_plan_fn = build_plan_fn
        self._execute_plan_fn = execute_plan_fn
        self._summarize_fn = summarize_fn
        self._module_data_path = Path(module_data_path)
        self._module_name = module_name

        self.prompt: Optional[str] = None
        self.brief: Optional[PromptStructure] = None
        self.plan: PlanStructure = PlanStructure()
        self.summary: Optional[str] = None
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None

    def build_instructions(self, prompt: str) -> None:
        """Return a concise brief for the project.

        Parameters
        ----------
        prompt : str
            The core request or goal for the project.

        Returns
        -------
        None
        """
        log("build_instructions", level=logging.INFO)
        self.start_date = datetime.utcnow()
        self.prompt = prompt
        self.brief = self._build_brief_fn(prompt)
        self.save()

    def build_plan(self) -> None:
        """Generate and store a structured plan based on the current brief.

        Raises
        ------
        ValueError
            If called before :meth:`build_instructions`.

        Returns
        -------
        None
        """
        log("build_plan", level=logging.INFO)
        if not self.brief:
            raise ValueError("Brief is required before building a plan.")

        plan = self._build_plan_fn(self.brief.prompt)
        if isinstance(plan, PlanStructure):
            self.plan = plan
        self.save()

    def execute_plan(self) -> List[str]:
        """Run each task, updating status, timestamps, and recorded results.

        Returns
        -------
        list[str]
            Flattened list of results from all executed tasks.
        """
        log("execute_plan", level=logging.INFO)
        if not self.plan:
            log("No tasks to execute.", level=logging.WARNING)
            return []

        compiled_results = self._execute_plan_fn(self.plan)
        self.save()
        return compiled_results

    def summarize_plan(self, results: Optional[List[str]] = None) -> str:
        """Summarize a collection of task outputs.

        Parameters
        ----------
        results : list[str], optional
            List of string outputs gathered from task execution. Defaults to
            ``None``, which uses the stored plan task results if available.

        Returns
        -------
        str
            Concise summary derived from the provided results.
        """
        log("summarize_plan", level=logging.INFO)

        if results is None:
            results = []
            if self.plan and self.plan.tasks:
                for task in self.plan.tasks:
                    results.extend(task.results or [])

        if not results:
            self.summary = ""
            return self.summary

        self.summary = self._summarize_fn(results)
        self.end_date = datetime.utcnow()
        self.save()
        return self.summary

    def run_plan(self, prompt: str) -> None:
        """Execute the full workflow for the provided prompt.

        Parameters
        ----------
        prompt : str
            The request or question to analyze and summarize.

        Returns
        -------
        None
        """
        self.build_instructions(prompt)
        self.build_plan()
        results = self.execute_plan()
        self.summarize_plan(results)

    @property
    def file_path(self) -> Path:
        """Return the path where the project snapshot will be stored.

        Returns
        -------
        Path
            Location of the JSON artifact for the current run.
        """
        if not self.start_date:
            self.start_date = datetime.utcnow()
        start_date_str = self.start_date.strftime(DATETIME_FMT)
        return self._module_data_path / self._module_name / f"{start_date_str}.json"

    def save(self) -> Path:
        """Persist the current project state to disk.

        Returns
        -------
        Path
            Path to the saved JSON artifact.
        """
        self.to_json_file(self.file_path)
        return self.file_path

    @staticmethod
    def _run_task(
        task: AgentTaskStructure,
        agent_callable: Callable[..., Any],
        aggregated_context: List[str],
    ) -> Any:
        """Execute a single task and return the raw result.

        Parameters
        ----------
        task : AgentTaskStructure
            Task definition containing the callable and inputs.
        aggregated_context : list[str]
            Context combined from the task and prior task outputs.

        Returns
        -------
        Any
            Raw output from the underlying callable.
        """
        task_type = ProjectManager._normalize_task_type(task.task_type)
        prompt_with_context = task.prompt
        if aggregated_context and task_type not in {"WebAgentSearch", "VectorSearch"}:
            context_block = "\n".join(aggregated_context)
            prompt_with_context = f"{task.prompt}\n\nContext:\n{context_block}"

        try:
            if task_type == "summarizer":
                summary_chunks: List[str] = [task.prompt] + aggregated_context
                output = agent_callable(summary_chunks)
            elif task_type in {"WebAgentSearch", "VectorSearch"}:
                output = agent_callable(task.prompt)
            else:
                output = agent_callable(
                    prompt_with_context,
                    context=aggregated_context,
                )
        except TypeError:
            output = agent_callable(prompt_with_context)
        except Exception as exc:  # pragma: no cover - defensive guard
            log(
                f"Task '{task.task_type}' encountered an error: {exc}",
                level=logging.ERROR,
            )
            return f"Task error: {exc}"
        return ProjectManager._resolve_result(output)

    @staticmethod
    def _run_task_in_thread(
        task: AgentTaskStructure,
        agent_callable: Callable[..., Any],
        aggregated_context: List[str],
    ) -> Any:
        """Execute a task in a background thread to avoid event-loop conflicts."""
        result_container: Dict[str, Any] = {"result": None, "error": None}

        def _runner() -> None:
            try:
                result_container["result"] = ProjectManager._run_task(
                    task,
                    agent_callable=agent_callable,
                    aggregated_context=aggregated_context,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                result_container["error"] = exc

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()
        if result_container["error"] is not None:
            raise result_container["error"]
        return result_container["result"]

    @staticmethod
    def _resolve_result(result: Any) -> Any:
        """Return awaited results when the callable is asynchronous.

        Parameters
        ----------
        result : Any
            Potentially awaitable output from a task callable.

        Returns
        -------
        Any
            Resolved output, awaited when necessary.
        """
        if not inspect.isawaitable(result):
            return result

        awaitable: asyncio.Future[Any] | asyncio.Task[Any] | Any = result
        coroutine = (
            awaitable
            if inspect.iscoroutine(awaitable)
            else ProjectManager._await_wrapper(awaitable)
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        if loop.is_running():
            resolved_result: Any = None

            def _run_in_thread() -> None:
                nonlocal resolved_result
                resolved_result = asyncio.run(coroutine)

            thread = threading.Thread(target=_run_in_thread, daemon=True)
            thread.start()
            thread.join()
            return resolved_result

        return loop.run_until_complete(coroutine)

    @staticmethod
    async def _await_wrapper(awaitable: Any) -> Any:
        """Await a generic awaitable and return its result."""
        return await awaitable

    @staticmethod
    def _normalize_results(result: Any) -> List[str]:
        """Convert agent outputs into a list of strings.

        Parameters
        ----------
        result : Any
            Raw output from a task execution.

        Returns
        -------
        list[str]
            Normalized string values representing the output.
        """
        if result is None:
            return []
        if isinstance(result, list):
            return [str(item) for item in result]
        return [str(result)]

    def _persist_task_results(self, task: AgentTaskStructure) -> Path:
        """Write task context and results to disk for future analysis."""
        run_dir = self._get_run_directory()
        task_label = self._task_label(task)
        file_path = run_dir / f"{task_label}.json"
        task.to_json_file(str(file_path))
        return file_path

    def _get_run_directory(self) -> Path:
        """Return (and create) the directory used to persist task artifacts."""
        if not hasattr(self, "_run_directory"):
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self._run_directory = (
                self._module_data_path
                / Path(self._module_name)
                / "project_manager"
                / timestamp
            )
        self._run_directory.mkdir(parents=True, exist_ok=True)
        return self._run_directory

    @staticmethod
    def _task_label(task: AgentTaskStructure) -> str:
        """Generate a filesystem-safe label for the task."""
        task_type = ProjectManager._normalize_task_type(task.task_type)
        base = (task_type or "task").replace(" ", "_").lower()
        return f"{base}_{task_type}"

    @staticmethod
    def _normalize_task_type(task_type: AgentEnum | str) -> str:
        """Return the normalized task type string."""
        if isinstance(task_type, AgentEnum):
            return task_type.value
        if task_type in AgentEnum.__members__:
            return AgentEnum.__members__[task_type].value
        try:
            return AgentEnum(task_type).value
        except ValueError:
            return str(task_type)


__all__ = ["ProjectManager"]

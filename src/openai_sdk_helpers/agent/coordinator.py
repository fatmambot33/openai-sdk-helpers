"""Generic project manager for coordinating agent plans."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


from ..structure import TaskStructure, PlanStructure, PromptStructure, AgentEnum
from ..utils import ensure_directory, log
from .base import AgentBase
from .configuration import AgentConfiguration

PromptFn = Callable[[str], PromptStructure]
BuildPlanFn = Callable[[str], PlanStructure]
ExecutePlanFn = Callable[[PlanStructure], List[str]]
SummarizeFn = Callable[[List[str]], str]


class CoordinatorAgent(AgentBase):
    """Coordinate agent plans while persisting project state and outputs.

    Parameters
    ----------
    prompt_fn : PromptFn
        Callable that generates a prompt brief from the input string.
    build_plan_fn : BuildPlanFn
        Callable that generates a plan from the prompt brief.
    execute_plan_fn : ExecutePlanFn
        Callable that executes a plan and returns results.
    summarize_fn : SummarizeFn
        Callable that summarizes a list of result strings.
    module_data_path : Path
        Base path for persisting project artifacts.
    name : str
        Name of the parent module for data organization.
    configuration : AgentConfiguration or None, default=None
        Optional agent configuration describing prompts and metadata.
    template_path : Path or None, default=None
        Optional template file path for prompt rendering.
    model : str or None, default=None
        Model identifier to use for coordinator operations.

    Methods
    -------
    build_prompt(prompt)
        Summarize the prompt into a concise brief.
    build_plan()
        Create a list of ``TaskStructure`` entries for the project.
    execute_plan()
        Run each task sequentially while tracking status and timing.
    summarize_plan(results)
        Summarize a collection of result strings.
    run_plan(prompt)
        Execute the prompt-to-summary workflow end to end.
    file_path
        Path to the JSON artifact for the current run.
    to_dict()
        Return a JSON-serializable snapshot of stored project data.
    save()
        Persist the stored project data to a JSON file.
    to_json()
        Return a JSON-compatible dict representation (inherited from JSONSerializable).
    to_json_file(filepath)
        Write serialized JSON data to a file path (inherited from JSONSerializable).
    from_json(data)
        Create an instance from a JSON-compatible dict (class method, inherited from JSONSerializable).
    from_json_file(filepath)
        Load an instance from a JSON file (class method, inherited from JSONSerializable).
    """

    def __init__(
        self,
        *,
        prompt_fn: PromptFn,
        build_plan_fn: BuildPlanFn,
        execute_plan_fn: ExecutePlanFn,
        summarize_fn: SummarizeFn,
        module_data_path: Path,
        name: str,
        configuration: Optional[AgentConfiguration] = None,
        template_path: Optional[Path] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the project manager with injected workflow helpers.

        Parameters
        ----------
        prompt_fn : PromptFn
            Callable that generates a prompt brief from the input string.
        build_plan_fn : BuildPlanFn
            Callable that generates a plan from the prompt brief.
        execute_plan_fn : ExecutePlanFn
            Callable that executes a plan and returns results.
        summarize_fn : SummarizeFn
            Callable that summarizes a list of result strings.
        module_data_path : Path
            Base path for persisting project artifacts.
        name : str
            Name of the parent module for data organization.
        configuration : AgentConfiguration or None, default=None
            Optional agent configuration describing prompts and metadata.
        template_path : Path or None, default=None
            Optional template file path for prompt rendering.
        model : str or None, default=None
            Model identifier to use for coordinator operations.

        Raises
        ------
        ValueError
            If the provided configuration is invalid.

        Examples
        --------
        >>> coordinator = CoordinatorAgent(
        ...     prompt_fn=lambda p: PromptStructure(prompt=p),
        ...     build_plan_fn=lambda p: PlanStructure(),
        ...     execute_plan_fn=lambda p: [],
        ...     summarize_fn=lambda r: "summary",
        ...     module_data_path=Path("."),
        ...     name="test",
        ... )
        """
        if configuration is None:
            configuration = AgentConfiguration(
                name=__class__.__name__,
                instructions="Coordinate agents for planning and summarization.",
                description="Coordinates agents for planning and summarization.",
                template_path=template_path,
                model=model,
            )
        super().__init__(configuration=configuration)
        self._prompt_fn = prompt_fn
        self._build_plan_fn = build_plan_fn
        self._execute_plan_fn = execute_plan_fn
        self._summarize_fn = summarize_fn
        self._module_data_path = Path(module_data_path)
        self._name = name

        self.prompt: Optional[str] = None
        self.brief: Optional[PromptStructure] = None
        self.plan: PlanStructure = PlanStructure()
        self.summary: Optional[str] = None
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None

    def build_prompt(self, prompt: str) -> None:
        """Generate a concise brief for the project.

        Parameters
        ----------
        prompt : str
            The core request or goal for the project.

        Examples
        --------
        >>> coordinator.build_prompt("Analyze the impact of AI on healthcare.")
        """
        log("build_prompt", level=logging.INFO)
        self.start_date = datetime.now(timezone.utc)
        self.prompt = prompt
        self.brief = self._prompt_fn(prompt)
        self.save()

    def build_plan(self) -> None:
        """Generate and store a structured plan based on the current brief.

        Raises
        ------
        ValueError
            If called before :meth:`build_prompt`.

        Examples
        --------
        >>> coordinator.build_prompt("Analyze AI in healthcare.")
        >>> coordinator.build_plan()
        """
        log("build_plan", level=logging.INFO)
        if not self.brief:
            raise ValueError("Brief is required before building a plan.")

        plan = self._build_plan_fn(self.brief.prompt)
        if isinstance(plan, PlanStructure):
            self.plan = plan
        self.save()

    def execute_plan(self) -> List[str]:
        """Execute each task, updating status, timestamps, and recorded results.

        Returns
        -------
        list[str]
            Flattened list of results from all executed tasks.

        Examples
        --------
        >>> coordinator.build_prompt("Analyze AI.")
        >>> coordinator.build_plan()
        >>> results = coordinator.execute_plan()
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
        results : list[str] or None, default=None
            List of string outputs gathered from task execution. When ``None``,
            uses the stored plan task results if available.

        Returns
        -------
        str
            Concise summary derived from the provided results.

        Examples
        --------
        >>> results = ["AI is impacting healthcare.", "New models are faster."]
        >>> summary = coordinator.summarize_plan(results)
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
        self.end_date = datetime.now(timezone.utc)
        self.save()
        return self.summary

    def run_plan(self, prompt: str) -> None:
        """Execute the full workflow for the provided prompt.

        Parameters
        ----------
        prompt : str
            The request or question to analyze and summarize.

        Examples
        --------
        >>> coordinator.run_plan("Analyze the future of AI.")
        """
        self.build_prompt(prompt)
        self.build_plan()
        results = self.execute_plan()
        self.summarize_plan(results)

    @staticmethod
    def _run_task(
        task: TaskStructure,
        agent_callable: Callable[..., Any],
        aggregated_context: List[str],
    ) -> Any:
        """Execute a single task and return the raw result.

        Parameters
        ----------
        task : TaskStructure
            Task definition containing the callable and inputs.
        agent_callable : Callable[..., Any]
            Callable that executes the task prompt and returns a result.
        aggregated_context : list[str]
            Context combined from the task and prior task outputs.

        Returns
        -------
        Any
            Raw output from the underlying callable.
        """
        task_type = CoordinatorAgent._normalize_task_type(task.task_type)
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
        return CoordinatorAgent._resolve_result(output)

    @staticmethod
    def _run_task_in_thread(
        task: TaskStructure,
        agent_callable: Callable[..., Any],
        aggregated_context: List[str],
    ) -> Any:
        """Execute a task in a background thread to avoid event-loop conflicts.

        Parameters
        ----------
        task : TaskStructure
            Task definition containing the callable and inputs.
        agent_callable : Callable[..., Any]
            Callable that executes the task prompt and returns a result.
        aggregated_context : list[str]
            Context combined from the task and prior task outputs.

        Returns
        -------
        Any
            Resolved output from the underlying callable.
        """
        result_container: Dict[str, Any] = {"result": None, "error": None}

        def _runner() -> None:
            try:
                result_container["result"] = CoordinatorAgent._run_task(
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

        if isinstance(result, (asyncio.Future, asyncio.Task)):
            if result.done():
                return result.result()

            try:
                owning_loop = result.get_loop()
            except AttributeError:  # pragma: no cover - defensive guard
                owning_loop = None
            if owning_loop is not None and owning_loop.is_running():
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None
                if current_loop is not None and current_loop is owning_loop:
                    raise RuntimeError(
                        "Cannot resolve a pending task from its owning running event loop; "
                        "await the task instead."
                    )
                return asyncio.run_coroutine_threadsafe(
                    CoordinatorAgent._await_wrapper(result), owning_loop
                ).result()

        awaitable: asyncio.Future[Any] | asyncio.Task[Any] | Any = result
        coroutine = (
            awaitable
            if inspect.iscoroutine(awaitable)
            else CoordinatorAgent._await_wrapper(awaitable)
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
        """Resolve an awaitable and return its result.

        Parameters
        ----------
        awaitable : Any
            Awaitable object to resolve.

        Returns
        -------
        Any
            Result of the awaited object.
        """
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

    def _persist_task_results(self, task: TaskStructure) -> Path:
        """Write task context and results to disk for future analysis.

        Parameters
        ----------
        task : TaskStructure
            Task definition containing the callable and inputs.

        Returns
        -------
        Path
            Location where the task artifact was saved.
        """
        run_dir = self._get_run_directory()
        task_label = self._task_label(task)
        file_path = run_dir / f"{task_label}.json"
        task.to_json_file(str(file_path))
        return file_path

    def _get_run_directory(self) -> Path:
        """Return (and create) the directory used to persist task artifacts.

        Returns
        -------
        Path
            Directory where task outputs are stored for the run.
        """
        if not hasattr(self, "_run_directory"):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._run_directory = (
                self._module_data_path
                / Path(self._name)
                / "coordinator_agent"
                / timestamp
            )
        return ensure_directory(self._run_directory)

    @staticmethod
    def _task_label(task: TaskStructure) -> str:
        """Generate a filesystem-safe label for the task.

        Parameters
        ----------
        task : TaskStructure
            Task definition containing the callable and inputs.

        Returns
        -------
        str
            Lowercase label safe for filesystem usage.
        """
        task_type = CoordinatorAgent._normalize_task_type(task.task_type)
        base = (task_type or "task").replace(" ", "_").lower()
        return f"{base}_{task_type}"

    @staticmethod
    def _normalize_task_type(task_type: AgentEnum | str) -> str:
        """Return the normalized task type string.

        Parameters
        ----------
        task_type : AgentEnum or str
            Task classification to normalize.

        Returns
        -------
        str
            String representation of the task type.
        """
        if isinstance(task_type, AgentEnum):
            return task_type.value
        if task_type in AgentEnum.__members__:
            return AgentEnum.__members__[task_type].value
        try:
            return AgentEnum(task_type).value
        except ValueError:
            return str(task_type)


__all__ = ["CoordinatorAgent"]

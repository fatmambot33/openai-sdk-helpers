"""Structured output model for agent plans.

This module defines a Pydantic model for representing ordered lists of agent
tasks, with support for sequential execution and result aggregation.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from datetime import datetime, timezone
from typing import Any, Awaitable, Coroutine, cast
from collections.abc import Mapping

from .enum import AgentEnum
from ..base import StructureBase, spec_field
from .task import TaskStructure
from .types import AgentCallable, AgentRegistry


class PlanStructure(StructureBase):
    """Structured representation of an ordered list of agent tasks.

    Represents a complete execution plan consisting of multiple agent tasks
    to be run sequentially, with support for context passing between tasks.

    Attributes
    ----------
    tasks : list[TaskStructure]
        Ordered list of agent tasks to execute.

    Methods
    -------
    print()
        Return a formatted description of every task in order.
    __len__()
        Return the count of tasks in the plan.
    append(task)
        Append a TaskStructure to the plan.
    execute(agent_registry, halt_on_error)
        Run tasks sequentially using the provided agent callables.

    Examples
    --------
    >>> plan = PlanStructure(tasks=[
    ...     TaskStructure(prompt="Task 1"),
    ...     TaskStructure(prompt="Task 2")
    ... ])
    >>> len(plan)
    2
    """

    tasks: list[TaskStructure] = spec_field(
        "tasks",
        default_factory=list,
        description="Ordered list of agent tasks to execute.",
    )

    def __len__(self) -> int:
        """Return the number of tasks in the plan.

        Returns
        -------
        int
            Count of stored agent tasks.

        Examples
        --------
        >>> len(PlanStructure())
        0
        """
        return len(self.tasks)

    def append(self, task: TaskStructure) -> None:
        """Add a task to the plan in execution order.

        Parameters
        ----------
        task : TaskStructure
            Task to append to the plan.

        Examples
        --------
        >>> plan = PlanStructure()
        >>> plan.append(TaskStructure(prompt="Test"))  # doctest: +SKIP
        """
        self.tasks.append(task)

    def execute(
        self,
        agent_registry: AgentRegistry,
        *,
        halt_on_error: bool = True,
    ) -> list[str]:
        """Execute tasks with registered agent callables and record outputs.

        Runs each task in sequence, passing results as context to subsequent
        tasks. Updates task status, timing, and results as execution proceeds.

        Parameters
        ----------
        agent_registry : AgentRegistry
            Lookup of agent identifiers to callables. Keys may be AgentEnum
            instances or their string values. Each callable receives the task
            prompt (augmented with prior context) and an optional context
            keyword containing accumulated results.
        halt_on_error : bool, default=True
            Whether execution should stop when a task raises an exception.

        Returns
        -------
        list[str]
            Flattened list of normalized outputs from executed tasks.

        Raises
        ------
        KeyError
            If a task does not have a corresponding callable in agent_registry.

        Examples
        --------
        >>> def agent_fn(prompt, context=None):
        ...     return f"Result for {prompt}"
        >>> registry = {AgentEnum.WEB_SEARCH: agent_fn}
        >>> plan = PlanStructure(tasks=[TaskStructure(prompt="Test")])
        >>> results = plan.execute(registry)  # doctest: +SKIP
        """
        normalized_registry: dict[str, AgentCallable] = {
            self._resolve_registry_key(key): value
            for key, value in agent_registry.items()
        }

        aggregated_results: list[str] = []
        for task in self.tasks:
            callable_key = self._resolve_registry_key(task.task_type)
            if callable_key not in normalized_registry:
                raise KeyError(f"No agent registered for '{callable_key}'.")

            agent_callable = normalized_registry[callable_key]
            task.start_date = datetime.now(timezone.utc)
            task.status = "running"

            try:
                result = self._run_task(
                    task,
                    agent_callable=agent_callable,
                    aggregated_context=list(aggregated_results),
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                task.status = "error"
                task.results = [f"Task error: {exc}"]
                task.end_date = datetime.now(timezone.utc)
                if halt_on_error:
                    break
                aggregated_results.extend(task.results)
                continue

            normalized = self._normalize_results(result)
            task.results = normalized
            aggregated_results.extend(normalized)
            task.status = "done"
            task.end_date = datetime.now(timezone.utc)

        return aggregated_results

    @staticmethod
    def _resolve_registry_key(task_type: AgentEnum | str) -> str:
        """Return a normalized registry key for the given task_type.

        Parameters
        ----------
        task_type : AgentEnum | str
            Task type to normalize.

        Returns
        -------
        str
            Normalized key for agent registry lookup.
        """
        if isinstance(task_type, AgentEnum):
            return task_type.value
        if task_type in AgentEnum.__members__:
            return AgentEnum.__members__[task_type].value
        try:
            return AgentEnum(task_type).value
        except ValueError:
            return str(task_type)

    @staticmethod
    def _run_task(
        task: TaskStructure,
        *,
        agent_callable: AgentCallable,
        aggregated_context: list[str],
    ) -> object | Coroutine[Any, Any, object]:
        """Execute a single task using the supplied callable.

        Combines task context with aggregated results from previous tasks,
        then invokes the agent callable with the augmented prompt.

        Parameters
        ----------
        task : TaskStructure
            Task definition containing inputs and metadata.
        agent_callable : AgentCallable
            Function responsible for performing the task.
        aggregated_context : list[str]
            Accumulated results from previously executed tasks.

        Returns
        -------
        Any
            Raw output from the callable.
        """
        task_context = list(task.context or [])
        combined_context = task_context + list(aggregated_context)

        prompt_with_context = task.prompt
        if combined_context:
            context_block = "\n".join(combined_context)
            prompt_with_context = f"{task.prompt}\n\nContext:\n{context_block}"

        try:
            return agent_callable(prompt_with_context, context=combined_context)
        except TypeError:
            return agent_callable(prompt_with_context)

    @staticmethod
    def _normalize_results(result: object | Coroutine[Any, Any, object]) -> list[str]:
        """Convert callable outputs into a list of strings.

        Handles various result types including None, awaitables, lists,
        and single values.

        Parameters
        ----------
        result : Any
            Raw result from agent callable.

        Returns
        -------
        list[str]
            Normalized list of string results.
        """
        if result is None:
            return []
        if inspect.isawaitable(result):
            awaited = PlanStructure._await_result(
                cast(Coroutine[Any, Any, object], result)
            )
            return PlanStructure._normalize_results(awaited)
        if isinstance(result, list):
            return [str(item) for item in result]
        return [str(result)]

    @staticmethod
    def _await_result(result: Coroutine[Any, Any, object]) -> object:
        """Await the provided result, handling running event loops.

        Properly handles awaiting results whether an event loop is running
        or not, using a separate thread when necessary.

        Parameters
        ----------
        result : Any
            Awaitable result to resolve.

        Returns
        -------
        Any
            Resolved value from the awaitable.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(result)

        if loop.is_running():
            container: dict[str, object | None] = {"value": None}

            def _runner() -> None:
                container["value"] = asyncio.run(result)

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            return container["value"]

        return loop.run_until_complete(result)


__all__ = ["PlanStructure"]

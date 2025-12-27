"""Structured output model for agent plans."""

from __future__ import annotations

import asyncio
import inspect
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping

from ..base import BaseStructure, spec_field
from .enum import AgentEnum
from .task import AgentTaskStructure


class PlanStructure(BaseStructure):
    """Structured representation of an ordered list of agent tasks.

    Methods
    -------
    print()
        Return a formatted description of every task in order.
    __len__()
        Return the count of tasks in the plan.
    append(task)
        Append an ``AgentTaskStructure`` to the plan.
    execute(agent_registry, halt_on_error)
        Run tasks sequentially using the provided agent callables.
    """

    tasks: List[AgentTaskStructure] = spec_field(
        "tasks",
        default_factory=list,
        description="Ordered list of agent tasks to execute.",
    )

    def print(self) -> str:
        """Return a human-readable representation of the plan.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Concatenated description of each plan step.

        Raises
        ------
        None

        Examples
        --------
        >>> PlanStructure().print()
        'No tasks defined.'
        """
        if not self.tasks:
            return "No tasks defined."
        return "\n\n".join(
            [f"Task {idx + 1}:\n{task.print()}" for idx, task in enumerate(self.tasks)]
        )

    def __len__(self) -> int:
        """Return the number of tasks contained in the plan.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Count of stored agent tasks.

        Raises
        ------
        None

        Examples
        --------
        >>> len(PlanStructure())
        0
        """
        return len(self.tasks)

    def append(self, task: AgentTaskStructure) -> None:
        """Add a task to the plan in execution order.

        Parameters
        ----------
        task : AgentTaskStructure
            Task to append to the plan.

        Returns
        -------
        None

        Raises
        ------
        None

        Examples
        --------
        >>> plan = PlanStructure()
        >>> plan.append(AgentTaskStructure(prompt="Test"))  # doctest: +SKIP
        """
        self.tasks.append(task)

    def execute(
        self,
        agent_registry: Mapping[AgentEnum | str, Callable[..., Any]],
        *,
        halt_on_error: bool = True,
    ) -> list[str]:
        """Execute tasks with registered agent callables and record outputs.

        Parameters
        ----------
        agent_registry : Mapping[AgentEnum | str, Callable[..., Any]]
            Lookup of agent identifiers to callables. Keys may be ``AgentEnum``
            instances or their string values. Each callable receives the task
            prompt (augmented with prior context) and an optional ``context``
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
            If a task does not have a corresponding callable in
            ``agent_registry``.
        """

        aggregated_results: list[str] = []
        for task in self.tasks:
            callable_key = self._resolve_registry_key(task.task_type)
            if callable_key not in agent_registry:
                raise KeyError(f"No agent registered for '{callable_key}'.")

            agent_callable = agent_registry[callable_key]
            task.start_date = datetime.utcnow()
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
                task.end_date = datetime.utcnow()
                if halt_on_error:
                    break
                aggregated_results.extend(task.results)
                continue

            normalized = self._normalize_results(result)
            task.results = normalized
            aggregated_results.extend(normalized)
            task.status = "done"
            task.end_date = datetime.utcnow()

        return aggregated_results

    @staticmethod
    def _resolve_registry_key(task_type: AgentEnum | str) -> str:
        """Return a normalized registry key for the given ``task_type``."""

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
        task: AgentTaskStructure,
        *,
        agent_callable: Callable[..., Any],
        aggregated_context: list[str],
    ) -> Any:
        """Execute a single task using the supplied callable.

        Parameters
        ----------
        task : AgentTaskStructure
            Task definition containing inputs and metadata.
        agent_callable : Callable[..., Any]
            Function responsible for performing the task.
        aggregated_context : list[str]
            Accumulated results from previously executed tasks.

        Returns
        -------
        Any
            Raw output from the callable.
        """

        prompt_with_context = task.prompt
        if aggregated_context:
            context_block = "\n".join(aggregated_context)
            prompt_with_context = f"{task.prompt}\n\nContext:\n{context_block}"

        try:
            return agent_callable(prompt_with_context, context=aggregated_context)
        except TypeError:
            return agent_callable(prompt_with_context)

    @staticmethod
    def _normalize_results(result: Any) -> list[str]:
        """Convert callable outputs into a list of strings."""

        if result is None:
            return []
        if inspect.isawaitable(result):
            return PlanStructure._normalize_results(PlanStructure._await_result(result))
        if isinstance(result, list):
            return [str(item) for item in result]
        return [str(result)]

    @staticmethod
    def _await_result(result: Any) -> Any:
        """Await the provided result, handling running event loops."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(result)

        if loop.is_running():
            container: Dict[str, Any] = {"value": None}

            def _runner() -> None:
                container["value"] = asyncio.run(result)

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            return container["value"]

        return loop.run_until_complete(result)


__all__ = ["PlanStructure"]

"""Structured output model for agent plans."""

from __future__ import annotations

from typing import List

from ..base import BaseStructure, spec_field
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


__all__ = ["PlanStructure"]

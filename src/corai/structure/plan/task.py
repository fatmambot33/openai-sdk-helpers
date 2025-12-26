"""Structured output model for agent tasks."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import field_validator

from ..base import BaseStructure, spec_field
from .enum import AgentEnum


class AgentTaskStructure(BaseStructure):
    """Structured representation of a single agent task.

    Methods
    -------
    print()
        Return a formatted multi-line description of the task.
    """

    task_type: AgentEnum = spec_field(
        "task_type",
        default=AgentEnum.WEB_SEARCH,
        description="Agent type responsible for executing the task.",
    )
    prompt: str = spec_field(
        "prompt",
        description="Input passed to the agent.",
        examples=["Research the latest trends in AI-assisted data analysis."],
    )
    context: List[str] | None = spec_field(
        "context",
        default_factory=list,
        description="Additional context forwarded to the agent callable.",
    )
    start_date: Optional[datetime] = spec_field(
        "start_date",
        default=None,
        description="Timestamp marking when the task started (UTC).",
    )
    end_date: Optional[datetime] = spec_field(
        "end_date",
        default=None,
        description="Timestamp marking when the task completed (UTC).",
    )
    status: Literal["waiting", "running", "done", "error"] = spec_field(
        "status",
        default="waiting",
        description="Current lifecycle state for the task.",
    )
    results: List[str] = spec_field(
        "results",
        default_factory=list,
        description="Normalized string outputs returned by the agent.",
    )

    @field_validator("task_type", mode="before")
    @classmethod
    def _coerce_task_type(cls, value: AgentEnum | str) -> AgentEnum:
        """Coerce string inputs into ``AgentEnum`` values.

        Parameters
        ----------
        value : AgentEnum | str
            Enum instance or enum value string.

        Returns
        -------
        AgentEnum
            Parsed enum instance.

        Raises
        ------
        ValueError
            If the value cannot be mapped to a valid enum member.

        Examples
        --------
        >>> AgentTaskStructure._coerce_task_type("WebAgentSearch")
        <AgentEnum.WEB_SEARCH: 'WebAgentSearch'>
        """
        if isinstance(value, AgentEnum):
            return value
        return AgentEnum(value)

    def print(self) -> str:
        """Return a human-readable representation of the task.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Multi-line description of the task metadata.

        Raises
        ------
        None

        Examples
        --------
        >>> AgentTaskStructure(prompt="Test").print()
        'Task type: ...'  # doctest: +SKIP
        """
        return "\n".join(
            [
                BaseStructure.format_output("Task type", self.task_type),
                BaseStructure.format_output("Prompt", self.prompt),
                BaseStructure.format_output("Context", self.context),
                BaseStructure.format_output("Status", self.status),
                BaseStructure.format_output("Start date", self.start_date),
                BaseStructure.format_output("End date", self.end_date),
                BaseStructure.format_output("Results", self.results),
            ]
        )


__all__ = ["AgentTaskStructure"]

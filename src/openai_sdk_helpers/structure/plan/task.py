"""Structured output model for agent tasks.

This module defines a Pydantic model for representing individual agent tasks
within a plan, including task type, inputs, status tracking, and results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import field_validator

from .enum import AgentEnum
from ..base import StructureBase, spec_field


class TaskStructure(StructureBase):
    """Structured representation of a single agent task.

    Represents one task in an agent execution plan, including its type,
    inputs, execution status, timing, and results.

    Attributes
    ----------
    task_type : AgentEnum
        Agent type responsible for executing the task.
    prompt : str
        Input passed to the agent.
    context : list[str] or None
        Additional context forwarded to the agent callable.
    start_date : datetime or None
        Timestamp marking when the task started (UTC).
    end_date : datetime or None
        Timestamp marking when the task completed (UTC).
    status : Literal["waiting", "running", "done", "error"]
        Current lifecycle state for the task.
    results : list[str]
        Normalized string outputs returned by the agent.

    Methods
    -------
    print()
        Return a formatted multi-line description of the task.

    Examples
    --------
    >>> task = TaskStructure(
    ...     task_type=AgentEnum.WEB_SEARCH,
    ...     prompt="Research AI trends",
    ...     status="waiting"
    ... )
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
    context: list[str] | None = spec_field(
        "context",
        default_factory=list,
        description="Additional context forwarded to the agent callable.",
    )
    start_date: datetime | None = spec_field(
        "start_date",
        default=None,
        description="Timestamp marking when the task started (UTC).",
    )
    end_date: datetime | None = spec_field(
        "end_date",
        default=None,
        description="Timestamp marking when the task completed (UTC).",
    )
    status: Literal["waiting", "running", "done", "error"] = spec_field(
        "status",
        default="waiting",
        description="Current lifecycle state for the task.",
    )
    results: list[str] = spec_field(
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
        >>> TaskStructure._coerce_task_type("WebAgentSearch")
        <AgentEnum.WEB_SEARCH: 'WebAgentSearch'>
        """
        if isinstance(value, AgentEnum):
            return value
        return AgentEnum(value)


__all__ = ["TaskStructure"]

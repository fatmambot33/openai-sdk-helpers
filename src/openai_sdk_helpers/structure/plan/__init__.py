"""Structured output models for agent tasks and plans.

This package provides Pydantic models for representing agent execution plans,
including task definitions, agent type enumerations, and plan structures with
sequential execution support.

Classes
-------
PlanStructure
    Ordered list of agent tasks with execution capabilities.
TaskStructure
    Individual agent task with status tracking and results.
AgentEnum
    Enumeration of available agent types.
"""

from __future__ import annotations

from .plan import PlanStructure
from .task import TaskStructure
from .enum import AgentEnum

__all__ = [
    "PlanStructure",
    "TaskStructure",
    "AgentEnum",
]

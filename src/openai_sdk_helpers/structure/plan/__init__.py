"""Structured output models for agent tasks and plans.

This package provides Pydantic models for representing agent execution plans,
including task definitions, agent type enumerations, and plan structures with
sequential execution support. Also includes helper functions for creating and
executing plans.

Classes
-------
PlanStructure
    Ordered list of agent tasks with execution capabilities.
TaskStructure
    Individual agent task with status tracking and results.
AgentEnum
    Enumeration of available agent types.

Functions
---------
create_plan
    Create a PlanStructure from a sequence of tasks.
execute_task
    Execute a single task with an agent callable.
execute_plan
    Execute a complete plan using registered agent callables.
"""

from __future__ import annotations

from .plan import PlanStructure
from .task import TaskStructure
from .enum import AgentEnum
from .helpers import create_plan, execute_task, execute_plan

__all__ = [
    "PlanStructure",
    "TaskStructure",
    "AgentEnum",
    "create_plan",
    "execute_task",
    "execute_plan",
]

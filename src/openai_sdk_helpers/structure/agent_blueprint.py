"""Agent design and planning structures.

This module provides structures for capturing agent requirements and
converting them into executable plans with validation and deployment steps.
"""

from __future__ import annotations

from .base import BaseStructure, spec_field
from .plan import PlanStructure, TaskStructure
from .plan.enum import AgentEnum


class AgentBlueprint(BaseStructure):
    """Capture requirements for creating a new agent.

    Defines the complete specification for an agent including mission,
    capabilities, constraints, tools, data sources, and deployment plans.
    Can be converted into an executable plan structure.

    Attributes
    ----------
    name : str
        Name of the agent to build.
    mission : str
        Primary goal or charter for the agent.
    capabilities : list[str]
        Core skills the agent must perform.
    constraints : list[str]
        Boundaries, policies, or limits the agent must honor.
    required_tools : list[str]
        External tools the agent must integrate.
    data_sources : list[str]
        Data inputs that inform the agent's work.
    evaluation_plan : list[str]
        Checks, tests, or metrics that validate the agent.
    rollout_plan : list[str]
        Deployment or launch steps for the agent.
    guardrails : list[str]
        Safety rules and governance requirements.
    notes : str or None
        Additional context that informs the build.

    Methods
    -------
    summary()
        Return a human-readable overview of the blueprint.
    build_plan()
        Convert the blueprint into an ordered PlanStructure.

    Examples
    --------
    >>> blueprint = AgentBlueprint(
    ...     name="ResearchCoordinator",
    ...     mission="Coordinate research sprint",
    ...     capabilities=["search", "summarize"],
    ...     constraints=["max 10 queries per run"]
    ... )
    >>> print(blueprint.summary())
    >>> plan = blueprint.build_plan()
    """

    name: str = spec_field(
        "name",
        allow_null=False,
        description="Name of the agent to build.",
        examples=["ResearchCoordinator", "EvaluationRouter"],
    )
    mission: str = spec_field(
        "mission",
        allow_null=False,
        description="Primary goal or charter for the agent.",
        examples=["Coordinate a research sprint", "Score model outputs"],
    )
    capabilities: list[str] = spec_field(
        "capabilities",
        default_factory=list,
        description="Core skills the agent must perform.",
    )
    constraints: list[str] = spec_field(
        "constraints",
        default_factory=list,
        description="Boundaries, policies, or limits the agent must honor.",
    )
    required_tools: list[str] = spec_field(
        "required_tools",
        default_factory=list,
        description="External tools the agent must integrate.",
    )
    data_sources: list[str] = spec_field(
        "data_sources",
        default_factory=list,
        description="Data inputs that inform the agent's work.",
    )
    evaluation_plan: list[str] = spec_field(
        "evaluation_plan",
        default_factory=list,
        description="Checks, tests, or metrics that validate the agent.",
    )
    rollout_plan: list[str] = spec_field(
        "rollout_plan",
        default_factory=list,
        description="Deployment or launch steps for the agent.",
    )
    guardrails: list[str] = spec_field(
        "guardrails",
        default_factory=list,
        description="Safety rules and governance requirements.",
    )
    notes: str | None = spec_field(
        "notes",
        description="Additional context that informs the build.",
    )

    def summary(self) -> str:
        """Return a human-readable overview of the blueprint.

        Formats all blueprint fields into a multi-line summary highlighting
        key requirements and specifications.

        Returns
        -------
        str
            Multi-line formatted description of blueprint fields.

        Examples
        --------
        >>> print(blueprint.summary())
        Agent name: ResearchCoordinator
        Mission: Coordinate research sprint
        Capabilities: search, summarize
        ...
        """

        def _format(label: str, values: list[str]) -> str:
            return f"{label}: " + (", ".join(values) if values else "None")

        lines = [
            f"Agent name: {self.name}",
            f"Mission: {self.mission}",
            _format("Capabilities", self.capabilities),
            _format("Constraints", self.constraints),
            _format("Required tools", self.required_tools),
            _format("Data sources", self.data_sources),
            _format("Guardrails", self.guardrails),
            _format("Evaluation", self.evaluation_plan),
            _format("Rollout", self.rollout_plan),
        ]

        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)

    def build_plan(self) -> PlanStructure:
        """Translate the blueprint into a structured execution plan.

        Converts the agent requirements into an ordered sequence of tasks
        representing the complete build lifecycle: planning, design,
        implementation, validation, evaluation, and deployment.

        Returns
        -------
        PlanStructure
            Ordered list of tasks with agent types and prompts.

        Examples
        --------
        >>> plan = blueprint.build_plan()
        >>> len(plan.tasks)
        6
        >>> plan.tasks[0].task_type
        <AgentEnum.PLANNER: 'MetaPlanner'>
        """
        tasks = [
            TaskStructure(
                task_type=AgentEnum.PLANNER,
                prompt=self._scope_prompt(),
                context=self.constraints,
            ),
            TaskStructure(
                task_type=AgentEnum.DESIGNER,
                prompt=self._design_prompt(),
                context=self.required_tools + self.data_sources,
            ),
            TaskStructure(
                task_type=AgentEnum.BUILDER,
                prompt=self._synthesis_prompt(),
                context=self.capabilities,
            ),
            TaskStructure(
                task_type=AgentEnum.VALIDATOR,
                prompt=self._validation_prompt(),
                context=self.guardrails,
            ),
            TaskStructure(
                task_type=AgentEnum.EVALUATOR,
                prompt=self._evaluation_prompt(),
                context=self.evaluation_plan,
            ),
            TaskStructure(
                task_type=AgentEnum.RELEASE_MANAGER,
                prompt=self._deployment_prompt(),
                context=self.rollout_plan,
            ),
        ]
        plan = PlanStructure(tasks=tasks)
        return plan

    def _scope_prompt(self) -> str:
        """Return a scoping prompt based on mission, constraints, and guardrails."""
        return "\n".join(
            [
                f"Mission: {self.mission}",
                self._bullet_block("Guardrails", self.guardrails),
                self._bullet_block("Constraints", self.constraints),
            ]
        )

    def _design_prompt(self) -> str:
        """Return a design prompt covering tools, data, and capabilities."""
        return "\n".join(
            [
                self._bullet_block("Capabilities", self.capabilities),
                self._bullet_block("Required tools", self.required_tools),
                self._bullet_block("Data sources", self.data_sources),
            ]
        )

    def _synthesis_prompt(self) -> str:
        """Return a build prompt focused on interfaces and prompts."""
        return "\n".join(
            [
                "Design system and developer prompts that cover:",
                self._bullet_block("Mission", [self.mission]),
                self._bullet_block(
                    "Capabilities to implement",
                    self.capabilities or ["Draft standard handlers"],
                ),
            ]
        )

    def _validation_prompt(self) -> str:
        """Return a prompt instructing validation of guardrails and behaviors."""
        return "\n".join(
            [
                "Create automated validation for:",
                self._bullet_block("Guardrails", self.guardrails),
                self._bullet_block("Constraints", self.constraints),
            ]
        )

    def _evaluation_prompt(self) -> str:
        """Return an evaluation prompt emphasizing tests and metrics."""
        return "\n".join(
            [
                "Run evaluation and red-team scenarios using:",
                self._bullet_block("Evaluation plan", self.evaluation_plan),
                self._bullet_block("Capabilities", self.capabilities),
            ]
        )

    def _deployment_prompt(self) -> str:
        """Return a deployment prompt capturing rollout and monitoring."""
        return "\n".join(
            [
                self._bullet_block("Rollout steps", self.rollout_plan),
                self._bullet_block(
                    "Launch checklist",
                    [
                        "Observability hooks enabled",
                        "Runbook prepared",
                        "Rollback and kill switches validated",
                    ],
                ),
            ]
        )

    @staticmethod
    def _bullet_block(label: str, items: list[str]) -> str:
        """Return a labeled bullet block for use in prompts."""
        if not items:
            return f"{label}: None"
        bullets = "\n".join(f"- {item}" for item in items)
        return f"{label}:\n{bullets}"


__all__ = ["AgentBlueprint"]

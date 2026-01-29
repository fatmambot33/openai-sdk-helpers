"""Agent task enumeration definitions.

This module defines enumerations for agent types that can be assigned to
tasks within an execution plan.
"""

from __future__ import annotations

from ...enums.base import CrosswalkJSONEnum


class AgentEnum(CrosswalkJSONEnum):
    """Enumeration of available agent types for task execution.

    Defines all supported agent types that can be assigned to tasks in
    a plan. Each enum value corresponds to a specific agent implementation.

    Attributes
    ----------
    WEB_SEARCH : str
        Web search agent for retrieving information from the internet.
    VECTOR_SEARCH : str
        Vector search agent for semantic similarity search.
    DATA_ANALYST : str
        Data analysis agent for processing and analyzing data.
    SUMMARIZER : str
        Summarization agent for condensing information.
    TRANSLATOR : str
        Translation agent for language conversion.
    VALIDATOR : str
        Validation agent for checking constraints and guardrails.
    CLASSIFIER : str
        Taxonomy classifier agent for structured label selection.
    PLANNER : str
        Meta-planning agent for generating execution plans.
    DESIGNER : str
        Agent design agent for creating agent specifications.
    BUILDER : str
        Agent builder for constructing agent implementations.
    EVALUATOR : str
        Evaluation agent for assessing outputs and performance.
    RELEASE_MANAGER : str
        Release management agent for deployment coordination.

    Methods
    -------
    CROSSWALK()
        Return the raw crosswalk data for this enum.

    Examples
    --------
    >>> agent_type = AgentEnum.WEB_SEARCH
    >>> print(agent_type.value)
    'WebAgentSearch'
    """

    WEB_SEARCH = "WebAgentSearch"
    VECTOR_SEARCH = "VectorSearch"
    DATA_ANALYST = "DataAnalyst"
    SUMMARIZER = "SummarizerAgent"
    TRANSLATOR = "TranslatorAgent"
    VALIDATOR = "ValidatorAgent"
    CLASSIFIER = "TaxonomyClassifierAgent"
    PLANNER = "MetaPlanner"
    DESIGNER = "AgentDesigner"
    BUILDER = "AgentBuilder"
    EVALUATOR = "EvaluationAgent"
    RELEASE_MANAGER = "ReleaseManager"

    @classmethod
    def CROSSWALK(cls) -> dict[str, dict[str, str]]:
        """Return the raw crosswalk data for this enum.

        Returns
        -------
        dict[str, dict[str, Any]]
            Crosswalk mapping keyed by enum member.

        Raises
        ------
        None

        Examples
        --------
        >>> AgentEnum.CROSSWALK()["WEB_SEARCH"]["value"]
        'WebAgentSearch'
        """
        return {
            "WEB_SEARCH": {"value": "WebAgentSearch"},
            "VECTOR_SEARCH": {"value": "VectorSearch"},
            "DATA_ANALYST": {"value": "DataAnalyst"},
            "SUMMARIZER": {"value": "SummarizerAgent"},
            "TRANSLATOR": {"value": "TranslatorAgent"},
            "VALIDATOR": {"value": "ValidatorAgent"},
            "CLASSIFIER": {"value": "TaxonomyClassifierAgent"},
            "PLANNER": {"value": "MetaPlanner"},
            "DESIGNER": {"value": "AgentDesigner"},
            "BUILDER": {"value": "AgentBuilder"},
            "EVALUATOR": {"value": "EvaluationAgent"},
            "RELEASE_MANAGER": {"value": "ReleaseManager"},
        }


__all__ = ["AgentEnum"]

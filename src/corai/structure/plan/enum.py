"""Agent task enumeration definitions."""

from __future__ import annotations

from typing import Any

from ...enums.base import CrosswalkJSONEnum


class AgentEnum(CrosswalkJSONEnum):
    """Auto-generated enumeration for AgentEnum.

    Methods
    -------
    CROSSWALK()
        Return the raw crosswalk data for this enum.
    """

    WEB_SEARCH = "WebAgentSearch"
    VECTOR_SEARCH = "VectorSearch"
    DATA_ANALYST = "DataAnalyst"

    @classmethod
    def CROSSWALK(cls) -> dict[str, dict[str, Any]]:
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
        }


__all__ = ["AgentEnum"]

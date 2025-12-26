"""Configuration helpers for ``BaseAgent``."""

from __future__ import annotations

from typing import Any, List, Optional, Type

from agents.model_settings import ModelSettings
from pydantic import BaseModel, ConfigDict, Field

from ..structure import BaseStructure


class AgentConfig(BaseStructure):
    """Configuration required to build a :class:`BaseAgent`.

    Methods
    -------
    print()
        Return a human readable representation of the configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(title="Agent Name", description="Unique name for the agent")
    description: Optional[str] = Field(
        default=None, title="Description", description="Short description of the agent"
    )
    model: Optional[str] = Field(
        default=None, title="Model", description="Model identifier to use"
    )
    template_path: Optional[str] = Field(
        default=None, title="Template Path", description="Path to the Jinja template"
    )
    input_type: Optional[Type[BaseModel]] = Field(
        default=None,
        title="Input Type",
        description="Pydantic model describing the agent input",
    )
    output_type: Optional[Type[Any]] = Field(
        default=None,
        title="Output Type",
        description="Type describing the agent output; commonly a Pydantic model or builtin like ``str``",
    )
    tools: Optional[List[Any]] = Field(
        default=None,
        title="Tools",
        description="Tools available to the agent",
    )
    model_settings: Optional[ModelSettings] = Field(
        default=None, title="Model Settings", description="Additional model settings"
    )

    def print(self) -> str:
        """Return a human readable representation.

        Returns
        -------
        str
            The agent's name.
        """
        return self.name


__all__ = ["AgentConfig"]

AgentConfig.model_rebuild()

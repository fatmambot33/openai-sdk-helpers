"""Shared configuration for OpenAI SDK usage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


class OpenAISettings(BaseModel):
    """Configuration helpers for constructing OpenAI clients.

    Methods
    -------
    from_env(dotenv_path, **overrides)
        Build settings from environment variables and optional overrides.
    client_kwargs()
        Return keyword arguments for ``OpenAI`` initialization.
    create_client()
        Instantiate an ``OpenAI`` client using the stored configuration.
    """

    model_config = ConfigDict(extra="ignore")

    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key used to authenticate requests. Defaults to ``OPENAI_API_KEY``"
            " from the environment."
        ),
    )
    org_id: Optional[str] = Field(
        default=None,
        description=(
            "Organization identifier applied to outgoing requests. Defaults to"
            " ``OPENAI_ORG_ID``."
        ),
    )
    project_id: Optional[str] = Field(
        default=None,
        description=(
            "Project identifier used for billing and resource scoping. Defaults to"
            " ``OPENAI_PROJECT_ID``."
        ),
    )
    base_url: Optional[str] = Field(
        default=None,
        description=(
            "Custom base URL for self-hosted or proxied deployments. Defaults to"
            " ``OPENAI_BASE_URL``."
        ),
    )
    default_model: Optional[str] = Field(
        default=None,
        description=(
            "Model name used when constructing agents if no model is explicitly"
            " provided. Defaults to ``OPENAI_MODEL``."
        ),
    )

    @classmethod
    def from_env(
        cls, dotenv_path: Optional[Path] = None, **overrides: Any
    ) -> "OpenAISettings":
        """Load settings from the environment and optional overrides.

        Parameters
        ----------
        dotenv_path : Path | None
            Path to a ``.env`` file to load before reading environment
            variables. Default ``None``.
        overrides : Any
            Keyword overrides applied on top of environment values.

        Returns
        -------
        OpenAISettings
        Settings instance populated from environment variables and overrides.
        """
        env_file_values: Mapping[str, Optional[str]]
        if dotenv_path is not None:
            env_file_values = dotenv_values(dotenv_path)
        else:
            env_file_values = dotenv_values()

        values: Dict[str, Optional[str]] = {
            "api_key": overrides.get("api_key")
            or env_file_values.get("OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            "org_id": overrides.get("org_id")
            or env_file_values.get("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORG_ID"),
            "project_id": overrides.get("project_id")
            or env_file_values.get("OPENAI_PROJECT_ID")
            or os.getenv("OPENAI_PROJECT_ID"),
            "base_url": overrides.get("base_url")
            or env_file_values.get("OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
            "default_model": overrides.get("default_model")
            or env_file_values.get("OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL"),
        }

        settings = cls(**values)
        if not settings.api_key:
            source_hint = (
                f" from {dotenv_path}"
                if dotenv_path is not None
                else " from environment"
            )
            raise ValueError(
                "OPENAI_API_KEY is required to configure the OpenAI client"
                f" and was not found{source_hint}."
            )

        return settings

    def client_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for constructing an ``OpenAI`` client.

        Returns
        -------
        dict
        Keyword arguments populated with available authentication and routing
        values.
        """
        kwargs: Dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.org_id:
            kwargs["organization"] = self.org_id
        if self.project_id:
            kwargs["project"] = self.project_id
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs

    def create_client(self) -> OpenAI:
        """Instantiate an ``OpenAI`` client using the stored configuration.

        Returns
        -------
        OpenAI
        Client initialized with ``client_kwargs``.
        """
        return OpenAI(**self.client_kwargs())


__all__ = ["OpenAISettings"]

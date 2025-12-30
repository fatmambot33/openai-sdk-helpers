"""Shared configuration for OpenAI SDK usage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from openai_sdk_helpers.utils import (
    coerce_dict,
    coerce_optional_float,
    coerce_optional_int,
)


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
    timeout: Optional[float] = Field(
        default=None,
        description=(
            "Request timeout in seconds applied to all OpenAI client calls."
            " Defaults to ``OPENAI_TIMEOUT``."
        ),
    )
    max_retries: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of automatic retries for transient failures."
            " Defaults to ``OPENAI_MAX_RETRIES``."
        ),
    )
    extra_client_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments forwarded to ``openai.OpenAI``. Use"
            " this for less common options like ``default_headers`` or"
            " ``http_client``."
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

        timeout_raw = (
            overrides.get("timeout")
            or env_file_values.get("OPENAI_TIMEOUT")
            or os.getenv("OPENAI_TIMEOUT")
        )
        max_retries_raw = (
            overrides.get("max_retries")
            or env_file_values.get("OPENAI_MAX_RETRIES")
            or os.getenv("OPENAI_MAX_RETRIES")
        )

        values: Dict[str, Any] = {
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
            "timeout": coerce_optional_float(timeout_raw),
            "max_retries": coerce_optional_int(max_retries_raw),
            "extra_client_kwargs": coerce_dict(overrides.get("extra_client_kwargs")),
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
        kwargs: Dict[str, Any] = dict(self.extra_client_kwargs)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.org_id:
            kwargs["organization"] = self.org_id
        if self.project_id:
            kwargs["project"] = self.project_id
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
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

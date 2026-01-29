"""Shared configuration for OpenAI SDK usage.

This module provides the OpenAISettings class for centralized management of
OpenAI client configuration, reading from environment variables and .env files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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

    This class centralizes OpenAI SDK configuration by reading from environment
    variables and optional `.env` files, enabling consistent client setup across
    your application.

    Examples
    --------
    Load settings from environment and create a client:

    >>> from openai_sdk_helpers import OpenAISettings
    >>> settings = OpenAISettings.from_env()
    >>> client = settings.create_client()

    Override specific settings:

    >>> settings = OpenAISettings.from_env(
    ...     default_model="gpt-4o",
    ...     timeout=60.0
    ... )

    Methods
    -------
    from_env(dotenv_path, **overrides)
        Build settings from environment variables and optional overrides.
    from_secrets(secrets, **overrides)
        Build settings from a secrets mapping and optional overrides.
    client_kwargs()
        Return keyword arguments for ``OpenAI`` initialization.
    create_client()
        Instantiate an ``OpenAI`` client using the stored configuration.
    """

    model_config = ConfigDict(extra="ignore")

    api_key: str | None = Field(
        default=None,
        description=(
            "API key used to authenticate requests. Defaults to OPENAI_API_KEY"
            " from the environment."
        ),
    )
    org_id: str | None = Field(
        default=None,
        description=(
            "Organization identifier applied to outgoing requests. Defaults to"
            " OPENAI_ORG_ID."
        ),
    )
    project_id: str | None = Field(
        default=None,
        description=(
            "Project identifier used for billing and resource scoping. Defaults to"
            " OPENAI_PROJECT_ID."
        ),
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Custom base URL for self-hosted or proxied deployments. Defaults to"
            " OPENAI_BASE_URL."
        ),
    )
    default_model: str | None = Field(
        default=None,
        description=(
            "Model name used when constructing agents if no model is explicitly"
            " provided. Defaults to OPENAI_MODEL."
        ),
    )
    timeout: float | None = Field(
        default=None,
        description=(
            "Request timeout in seconds applied to all OpenAI client calls."
            " Defaults to OPENAI_TIMEOUT."
        ),
    )
    max_retries: int | None = Field(
        default=None,
        description=(
            "Maximum number of automatic retries for transient failures."
            " Defaults to OPENAI_MAX_RETRIES."
        ),
    )
    extra_client_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments forwarded to openai.OpenAI. Use"
            " this for less common options like default_headers or"
            " http_client."
        ),
    )

    @classmethod
    def from_env(
        cls, *, dotenv_path: Path | None = None, **overrides: Any
    ) -> OpenAISettings:
        """Load settings from the environment and optional overrides.

        Reads configuration from environment variables and an optional .env
        file, with explicit overrides taking precedence.

        Parameters
        ----------
        dotenv_path : Path or None, optional
            Path to a .env file to load before reading environment
            variables, by default None.
        overrides : Any
            Keyword overrides applied on top of environment values.

        Returns
        -------
        OpenAISettings
            Settings instance populated from environment variables and overrides.

        Raises
        ------
        ValueError
            If OPENAI_API_KEY is not found in environment or dotenv file.
        """
        env_file_values: Mapping[str, str | None] = {}
        if dotenv_path is not None:
            env_file_values = dotenv_values(dotenv_path)

        def first_non_none(*candidates: Any) -> Any:
            for candidate in candidates:
                if candidate is not None:
                    return candidate
            return None

        def resolve_value(override_key: str, env_var: str) -> Any:
            if dotenv_path is not None:
                return first_non_none(
                    overrides.get(override_key),
                    env_file_values.get(env_var),
                )
            return first_non_none(
                overrides.get(override_key),
                os.getenv(env_var),
            )

        timeout_raw = resolve_value("timeout", "OPENAI_TIMEOUT")
        max_retries_raw = resolve_value("max_retries", "OPENAI_MAX_RETRIES")

        values: dict[str, Any] = {
            "api_key": resolve_value("api_key", "OPENAI_API_KEY"),
            "org_id": resolve_value("org_id", "OPENAI_ORG_ID"),
            "project_id": resolve_value("project_id", "OPENAI_PROJECT_ID"),
            "base_url": resolve_value("base_url", "OPENAI_BASE_URL"),
            "default_model": resolve_value("default_model", "OPENAI_MODEL"),
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

    @classmethod
    def from_secrets(
        cls,
        secrets: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> OpenAISettings:
        """Load settings from a secrets mapping and optional overrides.

        Parameters
        ----------
        secrets : Mapping[str, Any] or None, optional
            Mapping of secret values keyed by environment variable names.
            Defaults to environment variables.
        overrides : Any
            Keyword overrides applied on top of secret values.

        Returns
        -------
        OpenAISettings
            Settings instance populated from secret values and overrides.

        Raises
        ------
        ValueError
            If OPENAI_API_KEY is not found in the secrets mapping.
        """
        secret_values: Mapping[str, Any] = secrets or os.environ

        def first_non_none(*candidates: Any) -> Any:
            for candidate in candidates:
                if candidate is not None:
                    return candidate
            return None

        def resolve_value(override_key: str, secret_key: str) -> Any:
            return first_non_none(
                overrides.get(override_key),
                secret_values.get(secret_key),
            )

        timeout_raw = resolve_value("timeout", "OPENAI_TIMEOUT")
        max_retries_raw = resolve_value("max_retries", "OPENAI_MAX_RETRIES")

        values: dict[str, Any] = {
            "api_key": resolve_value("api_key", "OPENAI_API_KEY"),
            "org_id": resolve_value("org_id", "OPENAI_ORG_ID"),
            "project_id": resolve_value("project_id", "OPENAI_PROJECT_ID"),
            "base_url": resolve_value("base_url", "OPENAI_BASE_URL"),
            "default_model": resolve_value("default_model", "OPENAI_MODEL"),
            "timeout": coerce_optional_float(timeout_raw),
            "max_retries": coerce_optional_int(max_retries_raw),
            "extra_client_kwargs": coerce_dict(overrides.get("extra_client_kwargs")),
        }

        settings = cls(**values)
        if not settings.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to configure the OpenAI client"
                " and was not found in secrets."
            )

        return settings

    def client_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for constructing an OpenAI client.

        Builds a dictionary containing all configured authentication and
        routing parameters suitable for OpenAI client initialization.

        Returns
        -------
        dict[str, Any]
            Keyword arguments populated with available authentication and
            routing values.
        """
        kwargs: dict[str, Any] = dict(self.extra_client_kwargs)
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
        """Instantiate an OpenAI client using the stored configuration.

        Uses client_kwargs() to build the client with all configured
        authentication and routing parameters.

        Returns
        -------
        OpenAI
            Client initialized with the configured settings.
        """
        return OpenAI(**self.client_kwargs())


__all__ = ["OpenAISettings", "build_openai_settings"]


def build_openai_settings(
    api_key: str | None = None,
    org_id: str | None = None,
    project_id: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
    timeout: float | str | None = None,
    max_retries: int | str | None = None,
    dotenv_path: Path | None = None,
    **extra_kwargs: Any,
) -> OpenAISettings:
    """Build OpenAISettings with validation and clear errors.

    Parameters
    ----------
    api_key : str or None, default None
        API key for OpenAI authentication. If None, reads from OPENAI_API_KEY.
    org_id : str or None, default None
        Organization ID. If None, reads from OPENAI_ORG_ID.
    project_id : str or None, default None
        Project ID. If None, reads from OPENAI_PROJECT_ID.
    base_url : str or None, default None
        Base URL for API requests. If None, reads from OPENAI_BASE_URL.
    default_model : str or None, default None
        Default model name. If None, reads from OPENAI_MODEL.
    timeout : float, str, or None, default None
        Request timeout in seconds. If None, reads from OPENAI_TIMEOUT.
        Strings are parsed to float.
    max_retries : int, str, or None, default None
        Maximum retry attempts. If None, reads from OPENAI_MAX_RETRIES.
        Strings are parsed to int.
    dotenv_path : Path or None, default None
        Path to a .env file. If None, uses environment only.
    **extra_kwargs : Any
        Additional keyword arguments forwarded to ``extra_client_kwargs``.

    Returns
    -------
    OpenAISettings
        Configured settings instance.

    Raises
    ------
    ValueError
        If required values are missing or cannot be parsed.
    TypeError
        If timeout or max_retries have invalid types.
    """
    parsed_timeout: float | None = None
    if timeout is not None:
        try:
            parsed_timeout = coerce_optional_float(timeout)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid timeout value '{timeout}'. Must be a number or numeric string."
            ) from exc

    parsed_max_retries: int | None = None
    if max_retries is not None:
        try:
            parsed_max_retries = coerce_optional_int(max_retries)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid max_retries value '{max_retries}'. "
                "Must be an integer or numeric string."
            ) from exc

    overrides = {}
    if api_key is not None:
        overrides["api_key"] = api_key
    if org_id is not None:
        overrides["org_id"] = org_id
    if project_id is not None:
        overrides["project_id"] = project_id
    if base_url is not None:
        overrides["base_url"] = base_url
    if default_model is not None:
        overrides["default_model"] = default_model
    if parsed_timeout is not None:
        overrides["timeout"] = parsed_timeout
    if parsed_max_retries is not None:
        overrides["max_retries"] = parsed_max_retries
    if extra_kwargs:
        overrides["extra_client_kwargs"] = extra_kwargs

    try:
        return OpenAISettings.from_env(dotenv_path=dotenv_path, **overrides)
    except ValueError as exc:
        raise ValueError(f"Failed to build OpenAI settings: {exc}") from exc

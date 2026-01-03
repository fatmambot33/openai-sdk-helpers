"""Configuration management for Streamlit chat applications.

This module provides Pydantic-based configuration validation and loading for
Streamlit chat applications. It handles response instantiation, vector store
attachment, and validation of application settings.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence, cast
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from openai_sdk_helpers.response.base import BaseResponse
from openai_sdk_helpers.structure.base import BaseStructure
from openai_sdk_helpers.utils import ensure_list


class StreamlitAppConfig(BaseModel):
    """Validated configuration for Streamlit chat applications.

    Manages all settings required to run a configuration-driven Streamlit
    chat interface, including response handlers, vector stores, display
    settings, and validation rules. Uses Pydantic for comprehensive
    validation and type safety.

    Attributes
    ----------
    response : BaseResponse, type[BaseResponse], Callable, or None
        Response handler as an instance, class, or callable factory.
    display_title : str
        Title displayed at the top of the Streamlit page.
    description : str or None
        Optional description shown beneath the title.
    system_vector_store : list[str] or None
        Optional vector store names to attach for file search.
    preserve_vector_stores : bool
        When True, skip automatic cleanup of vector stores on session close.
    model : str or None
        Optional model identifier displayed in the chat interface.

    Methods
    -------
    normalized_vector_stores()
        Return configured system vector stores as a list.
    create_response()
        Instantiate and return the configured BaseResponse.
    load_app_config(config_path)
        Load, validate, and return configuration from a Python module.

    Examples
    --------
    >>> from openai_sdk_helpers.streamlit_app import StreamlitAppConfig
    >>> config = StreamlitAppConfig(
    ...     response=MyResponse,
    ...     display_title="My Assistant",
    ...     description="A helpful AI assistant"
    ... )
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    response: BaseResponse[BaseStructure] | type[BaseResponse] | Callable | None = (
        Field(
            default=None,
            description=(
                "Configured ``BaseResponse`` subclass, instance, or callable that returns"
                " a response instance."
            ),
        )
    )
    display_title: str = Field(
        default="Example copilot",
        description="Title displayed at the top of the Streamlit page.",
    )
    description: str | None = Field(
        default=None,
        description="Optional short description shown beneath the title.",
    )
    system_vector_store: list[str] | None = Field(
        default=None,
        description=(
            "Optional vector store names to attach as system context for "
            "file search tools."
        ),
    )
    preserve_vector_stores: bool = Field(
        default=False,
        description="When ``True``, skip automatic vector store cleanup on close.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model hint for display alongside the chat interface.",
    )

    @field_validator("system_vector_store", mode="before")
    @classmethod
    def validate_vector_store(
        cls, value: Sequence[str] | str | None
    ) -> list[str] | None:
        """Normalize configured vector stores to a list of strings.

        Ensures that vector store configurations are always represented as
        a list, whether provided as a single string or sequence.

        Parameters
        ----------
        value : Sequence[str], str, or None
            Raw value from configuration (single name, list, or None).

        Returns
        -------
        list[str] or None
            Normalized list of vector store names, or None if not configured.

        Raises
        ------
        ValueError
            If any entry cannot be converted to a string.
        """
        if value is None:
            return None
        stores = ensure_list(value)
        if not all(isinstance(store, str) for store in stores):
            raise ValueError("system_vector_store values must be strings.")
        return list(stores)

    @field_validator("response")
    @classmethod
    def validate_response(
        cls, value: BaseResponse[BaseStructure] | type[BaseResponse] | Callable | None
    ) -> BaseResponse[BaseStructure] | type[BaseResponse] | Callable | None:
        """Validate that the response field is a valid handler source.

        Ensures the provided response can be used to create a BaseResponse
        instance for handling chat interactions.

        Parameters
        ----------
        value : BaseResponse, type[BaseResponse], Callable, or None
            Response handler as instance, class, or factory function.

        Returns
        -------
        BaseResponse, type[BaseResponse], Callable, or None
            Validated response handler.

        Raises
        ------
        TypeError
            If value is not a BaseResponse, subclass, or callable.
        """
        if value is None:
            return None
        if isinstance(value, BaseResponse):
            return value
        if isinstance(value, type) and issubclass(value, BaseResponse):
            return value
        if callable(value):
            return value
        raise TypeError("response must be a BaseResponse, subclass, or callable")

    def normalized_vector_stores(self) -> list[str]:
        """Return configured system vector stores as a list.

        Provides a consistent interface for accessing vector store names,
        returning an empty list when none are configured.

        Returns
        -------
        list[str]
            Vector store names, or empty list if not configured.

        Examples
        --------
        >>> config.normalized_vector_stores()
        ['docs', 'knowledge_base']
        """
        return list(self.system_vector_store or [])

    @model_validator(mode="after")
    def ensure_response(self) -> StreamlitAppConfig:
        """Validate that a response source is provided.

        Ensures the configuration includes a valid response handler, which
        is required for the chat application to function.

        Returns
        -------
        StreamlitAppConfig
            Self reference after validation.

        Raises
        ------
        ValueError
            If no response source is configured.
        """
        if self.response is None:
            raise ValueError("response must be provided.")
        return self

    def create_response(self) -> BaseResponse[BaseStructure]:
        """Instantiate and return the configured response handler.

        Converts the response field (whether class, instance, or callable)
        into an active BaseResponse instance ready for chat interactions.

        Returns
        -------
        BaseResponse[BaseStructure]
            Active response instance for handling chat messages.

        Raises
        ------
        TypeError
            If the configured response cannot produce a BaseResponse.

        Examples
        --------
        >>> response = config.create_response()
        >>> result = response.run_sync("Hello")
        """
        return _instantiate_response(self.response)

    @staticmethod
    def load_app_config(
        config_path: Path,
    ) -> StreamlitAppConfig:
        """Load, validate, and return configuration from a Python module.

        Imports the specified Python module and extracts its APP_CONFIG
        variable to create a validated StreamlitAppConfig instance.

        Parameters
        ----------
        config_path : Path
            Filesystem path to the Python configuration module.

        Returns
        -------
        StreamlitAppConfig
            Validated configuration extracted from the module.

        Raises
        ------
        FileNotFoundError
            If config_path does not exist.
        ImportError
            If the module cannot be imported.
        ValueError
            If APP_CONFIG is missing from the module.
        TypeError
            If APP_CONFIG has an invalid type.

        Examples
        --------
        >>> from pathlib import Path
        >>> config = StreamlitAppConfig.load_app_config(
        ...     Path("./my_config.py")
        ... )
        """
        module = _import_config_module(config_path)
        return _extract_config(module)


def _import_config_module(config_path: Path) -> ModuleType:
    """Import a Python module from the specified filesystem path.

    Uses importlib to dynamically load a configuration module, enabling
    runtime configuration discovery.

    Parameters
    ----------
    config_path : Path
        Filesystem path pointing to the configuration Python file.

    Returns
    -------
    ModuleType
        Loaded Python module containing application configuration.

    Raises
    ------
    FileNotFoundError
        If config_path does not exist on the filesystem.
    ImportError
        If the module cannot be imported or executed.

    Examples
    --------
    >>> module = _import_config_module(Path("./config.py"))
    >>> hasattr(module, 'APP_CONFIG')
    True
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'.")

    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load configuration module at '{config_path}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_config(module: ModuleType) -> StreamlitAppConfig:
    """Extract and validate StreamlitAppConfig from a loaded module.

    Looks for APP_CONFIG in the module and converts it to a validated
    StreamlitAppConfig instance. Supports multiple input formats including
    dictionaries, BaseResponse instances, and existing config objects.

    Parameters
    ----------
    module : ModuleType
        Python module loaded from the configuration path.

    Returns
    -------
    StreamlitAppConfig
        Parsed and validated configuration instance.

    Raises
    ------
    ValueError
        If APP_CONFIG is missing from the module.
    TypeError
        If APP_CONFIG is not a valid type (dict, BaseResponse, callable,
        or StreamlitAppConfig).

    Examples
    --------
    >>> config = _extract_config(module)
    >>> isinstance(config, StreamlitAppConfig)
    True
    """
    if not hasattr(module, "APP_CONFIG"):
        raise ValueError("APP_CONFIG must be defined in the configuration module.")

    raw_config = getattr(module, "APP_CONFIG")
    if isinstance(raw_config, StreamlitAppConfig):
        return raw_config
    if isinstance(raw_config, dict):
        return _config_from_mapping(raw_config)
    if isinstance(raw_config, BaseResponse):
        return StreamlitAppConfig(response=raw_config)
    if isinstance(raw_config, type) and issubclass(raw_config, BaseResponse):
        return StreamlitAppConfig(response=raw_config)
    if callable(raw_config):
        return StreamlitAppConfig(response=raw_config)

    raise TypeError(
        "APP_CONFIG must be a dict, callable, BaseResponse, or StreamlitAppConfig."
    )


def _instantiate_response(candidate: object) -> BaseResponse[BaseStructure]:
    """Convert a response candidate into a BaseResponse instance.

    Handles multiple candidate types: existing instances (returned as-is),
    classes (instantiated with no arguments), and callables (invoked to
    produce an instance).

    Parameters
    ----------
    candidate : object
        Response source as instance, class, or callable factory.

    Returns
    -------
    BaseResponse[BaseStructure]
        Active response instance ready for use.

    Raises
    ------
    TypeError
        If candidate cannot produce a BaseResponse instance.

    Examples
    --------
    >>> response = _instantiate_response(MyResponse)
    >>> isinstance(response, BaseResponse)
    True
    """
    if isinstance(candidate, BaseResponse):
        return candidate
    if isinstance(candidate, type) and issubclass(candidate, BaseResponse):
        response_cls = cast(type[BaseResponse[BaseStructure]], candidate)
        return response_cls()  # type: ignore[call-arg]
    if callable(candidate):
        response_callable = cast(Callable[[], BaseResponse[BaseStructure]], candidate)
        response = response_callable()
        if isinstance(response, BaseResponse):
            return response
    raise TypeError("response must be a BaseResponse, subclass, or callable")


def _config_from_mapping(raw_config: dict) -> StreamlitAppConfig:
    """Build StreamlitAppConfig from a dictionary with field aliases.

    Supports both 'response' and 'build_response' keys for backward
    compatibility. Extracts configuration fields and constructs a
    validated StreamlitAppConfig instance.

    Parameters
    ----------
    raw_config : dict
        Developer-supplied dictionary from the configuration module.

    Returns
    -------
    StreamlitAppConfig
        Validated configuration constructed from the dictionary.

    Examples
    --------
    >>> config = _config_from_mapping({
    ...     'response': MyResponse,
    ...     'display_title': 'My App'
    ... })
    """
    config_kwargs = dict(raw_config)
    response_candidate = config_kwargs.pop("response", None)
    if response_candidate is None:
        response_candidate = config_kwargs.pop("build_response", None)
    if response_candidate is not None:
        config_kwargs["response"] = response_candidate

    return StreamlitAppConfig(**config_kwargs)


def load_app_config(
    config_path: Path,
) -> StreamlitAppConfig:
    """Load and validate Streamlit configuration from a Python module.

    Convenience function that proxies to StreamlitAppConfig.load_app_config
    for backward compatibility.

    Parameters
    ----------
    config_path : Path
        Filesystem path to the configuration module.

    Returns
    -------
    StreamlitAppConfig
        Validated configuration loaded from the module.

    Examples
    --------
    >>> from pathlib import Path
    >>> config = load_app_config(Path("./my_config.py"))
    """
    return StreamlitAppConfig.load_app_config(config_path=config_path)


def _load_configuration(config_path: Path) -> StreamlitAppConfig:
    """Load configuration with user-friendly error handling for Streamlit.

    Wraps StreamlitAppConfig.load_app_config with exception handling that
    displays errors in the Streamlit UI and halts execution gracefully.

    Parameters
    ----------
    config_path : Path
        Filesystem location of the configuration module.

    Returns
    -------
    StreamlitAppConfig
        Validated configuration object.

    Raises
    ------
    RuntimeError
        If configuration loading fails (after displaying error in UI).

    Notes
    -----
    This function is designed specifically for use within Streamlit
    applications where errors should be displayed in the UI rather
    than raising exceptions that crash the app.
    """
    try:
        return StreamlitAppConfig.load_app_config(config_path=config_path)
    except Exception as exc:  # pragma: no cover - surfaced in UI
        import streamlit as st  # type: ignore[import-not-found]

        st.error(f"Configuration error: {exc}")
        st.stop()
        raise RuntimeError("Configuration loading halted.") from exc


__all__ = [
    "StreamlitAppConfig",
    "load_app_config",
    "_load_configuration",
]

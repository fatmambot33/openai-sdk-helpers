"""Configuration loading for the example Streamlit chat app."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence, cast
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure
from openai_sdk_helpers.utils import ensure_list

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "streamlit_app_config.py"


class StreamlitAppConfig(BaseModel):
    """Validated configuration for the config-driven Streamlit application.

    Methods
    -------
    normalized_vector_stores()
        Return configured system vector stores as a list of names.
    create_response()
        Instantiate the configured ``ResponseBase``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    response: ResponseBase[BaseStructure] | type[ResponseBase] | Callable | None = (
        Field(
            default=None,
            description=(
                "Configured ``ResponseBase`` subclass, instance, or callable that returns"
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
        """Normalize configured vector stores to a list of names.

        Parameters
        ----------
        value : Sequence[str] | str | None
            Raw value provided by the configuration module.

        Returns
        -------
        list[str] | None
            Normalized list of vector store names.

        Raises
        ------
        TypeError
            If any entry cannot be coerced to ``str``.
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
        cls, value: ResponseBase[BaseStructure] | type[ResponseBase] | Callable | None
    ) -> ResponseBase[BaseStructure] | type[ResponseBase] | Callable | None:
        """Ensure the configuration provides a valid response source."""
        if value is None:
            return None
        if isinstance(value, ResponseBase):
            return value
        if isinstance(value, type) and issubclass(value, ResponseBase):
            return value
        if callable(value):
            return value
        raise TypeError("response must be a ResponseBase, subclass, or callable")

    def normalized_vector_stores(self) -> list[str]:
        """Return configured system vector stores as a list.

        Returns
        -------
        list[str]
            Vector store names or an empty list when none are configured.
        """
        return list(self.system_vector_store or [])

    @model_validator(mode="after")
    def ensure_response(self) -> "StreamlitAppConfig":
        """Validate that a response source is provided."""
        if self.response is None:
            raise ValueError("response must be provided.")
        return self

    def create_response(self) -> ResponseBase[BaseStructure]:
        """Instantiate and return the configured response instance.

        Returns
        -------
        ResponseBase[BaseStructure]
            Active response instance.

        Raises
        ------
        TypeError
            If the configured ``response`` cannot produce a ``ResponseBase``.
        """
        return _instantiate_response(self.response)

    @staticmethod
    def load_app_config(
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> "StreamlitAppConfig":
        """Load, validate, and return the Streamlit application configuration.

        Parameters
        ----------
        config_path : Path, default=DEFAULT_CONFIG_PATH
            Filesystem path to the configuration module.

        Returns
        -------
        StreamlitAppConfig
            Validated configuration derived from ``config_path``.
        """
        module = _import_config_module(config_path)
        return _extract_config(module)


def _import_config_module(config_path: Path) -> ModuleType:
    """Import the configuration module from ``config_path``.

    Parameters
    ----------
    config_path : Path
        Filesystem path pointing to the configuration module.

    Returns
    -------
    ModuleType
        Loaded Python module containing application configuration.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ImportError
        If the module cannot be imported.
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
    """Extract a validated :class:`StreamlitAppConfig` from ``module``.

    Parameters
    ----------
    module : ModuleType
        Module loaded from the configuration path.

    Returns
    -------
    StreamlitAppConfig
        Parsed and validated configuration instance.

    Raises
    ------
    ValueError
        If ``APP_CONFIG`` is missing from the module.
    TypeError
        If ``APP_CONFIG`` is neither a mapping nor ``StreamlitAppConfig`` instance.
    """
    if not hasattr(module, "APP_CONFIG"):
        raise ValueError("APP_CONFIG must be defined in the configuration module.")

    raw_config = getattr(module, "APP_CONFIG")
    if isinstance(raw_config, StreamlitAppConfig):
        return raw_config
    if isinstance(raw_config, dict):
        return _config_from_mapping(raw_config)
    if isinstance(raw_config, ResponseBase):
        return StreamlitAppConfig(response=raw_config)
    if isinstance(raw_config, type) and issubclass(raw_config, ResponseBase):
        return StreamlitAppConfig(response=raw_config)
    if callable(raw_config):
        return StreamlitAppConfig(response=raw_config)

    raise TypeError(
        "APP_CONFIG must be a dict, callable, ResponseBase, or StreamlitAppConfig."
    )


def _instantiate_response(candidate: object) -> ResponseBase[BaseStructure]:
    """Instantiate a :class:`ResponseBase` from the provided candidate.

    Parameters
    ----------
    candidate : object
        Configured response source.

    Returns
    -------
    ResponseBase[BaseStructure]
        Active response instance.

    Raises
    ------
    TypeError
        If the candidate cannot produce a ``ResponseBase`` instance.
    """
    if isinstance(candidate, ResponseBase):
        return candidate
    if isinstance(candidate, type) and issubclass(candidate, ResponseBase):
        response_cls = cast(type[ResponseBase[BaseStructure]], candidate)
        return response_cls()  # type: ignore[call-arg]
    if callable(candidate):
        response_callable = cast(Callable[[], ResponseBase[BaseStructure]], candidate)
        response = response_callable()
        if isinstance(response, ResponseBase):
            return response
    raise TypeError("response must be a ResponseBase, subclass, or callable")


def _config_from_mapping(raw_config: dict) -> StreamlitAppConfig:
    """Build :class:`StreamlitAppConfig` from a mapping with aliases.

    The mapping may provide ``build_response`` directly or a ``response`` key
    containing a :class:`ResponseBase` instance, subclass, or callable.

    Parameters
    ----------
    raw_config : dict
        Developer-supplied mapping from the configuration module.

    Returns
    -------
    StreamlitAppConfig
        Validated configuration derived from ``raw_config``.
    """
    config_kwargs = dict(raw_config)
    response_candidate = config_kwargs.pop("response", None)
    if response_candidate is None:
        response_candidate = config_kwargs.pop("build_response", None)
    if response_candidate is not None:
        config_kwargs["response"] = response_candidate

    return StreamlitAppConfig(**config_kwargs)


def load_app_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> StreamlitAppConfig:
    """Proxy to :meth:`StreamlitAppConfig.load_app_config` for compatibility."""
    return StreamlitAppConfig.load_app_config(config_path=config_path)


def _load_configuration(config_path: Path = DEFAULT_CONFIG_PATH) -> StreamlitAppConfig:
    """Load the Streamlit configuration and present user-friendly errors.

    Parameters
    ----------
    config_path : Path, default=DEFAULT_CONFIG_PATH
        Filesystem location of the developer-authored configuration module.

    Returns
    -------
    StreamlitAppConfig
        Validated configuration object.
    """
    try:
        return StreamlitAppConfig.load_app_config(config_path=config_path)
    except Exception as exc:  # pragma: no cover - surfaced in UI
        import streamlit as st  # type: ignore[import-not-found]

        st.error(f"Configuration error: {exc}")
        st.stop()
        raise RuntimeError("Configuration loading halted.") from exc


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "StreamlitAppConfig",
    "load_app_config",
    "_load_configuration",
]

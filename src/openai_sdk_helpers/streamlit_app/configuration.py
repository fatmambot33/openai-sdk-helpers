"""Configuration loading for the example Streamlit chat app."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence
from pydantic import BaseModel, ConfigDict, Field, field_validator

from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure
from openai_sdk_helpers.utils import ensure_list

ResponseFactory = Callable[[], ResponseBase[BaseStructure]]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "streamlit_app_config.py"


class StreamlitAppConfig(BaseModel):
    """Validated configuration for the config-driven Streamlit application.

    Methods
    -------
    normalized_vector_stores()
        Return configured system vector stores as a list of names.
    """

    model_config = ConfigDict(extra="forbid")

    build_response: ResponseFactory = Field(
        description=(
            "Callable that constructs and returns a preconfigured ``ResponseBase`` "
            "instance."
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

    @field_validator("build_response")
    @classmethod
    def validate_builder(cls, value: ResponseFactory) -> ResponseFactory:
        """Ensure the configuration provides a callable response builder.

        Parameters
        ----------
        value : ResponseFactory
            Candidate response factory supplied by the configuration file.

        Returns
        -------
        ResponseFactory
            The original callable when validation succeeds.

        Raises
        ------
        TypeError
            If ``value`` is not callable.
        """

        if not callable(value):
            raise TypeError("build_response must be callable.")
        return value

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

    def normalized_vector_stores(self) -> list[str]:
        """Return configured system vector stores as a list.

        Returns
        -------
        list[str]
            Vector store names or an empty list when none are configured.
        """

        return list(self.system_vector_store or [])

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
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'."
        )

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
        return StreamlitAppConfig(build_response=lambda: raw_config)
    if isinstance(raw_config, type) and issubclass(raw_config, ResponseBase):
        return raw_config.build_streamlit_config()  # type: ignore[return-value]
    if callable(raw_config):
        return StreamlitAppConfig(build_response=_coerce_response_builder(raw_config))

    raise TypeError(
        "APP_CONFIG must be a dict, callable, ResponseBase, or StreamlitAppConfig."
    )


def _coerce_response_builder(candidate: object) -> ResponseFactory:
    """Convert a response candidate into a builder callable.

    Parameters
    ----------
    candidate : object
        Input from developer configuration representing the response session.

    Returns
    -------
    ResponseFactory
        Callable that yields a configured :class:`ResponseBase` instance.

    Raises
    ------
    TypeError
        If ``candidate`` cannot be turned into a response factory.
    """

    if isinstance(candidate, ResponseBase):
        return lambda: candidate
    if isinstance(candidate, type) and issubclass(candidate, ResponseBase):
        return lambda: candidate()
    if callable(candidate):
        return candidate
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
    if "response" in config_kwargs and "build_response" not in config_kwargs:
        response_candidate = config_kwargs.pop("response")
        config_kwargs["build_response"] = _coerce_response_builder(
            response_candidate
        )

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
        import streamlit as st

        st.error(f"Configuration error: {exc}")
        st.stop()


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "StreamlitAppConfig",
    "load_app_config",
    "_load_configuration",
]

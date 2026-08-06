"""Shared agent helpers built on the OpenAI Agents SDK."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from .._optional import import_optional_module
from ..structure.plan.enum import AgentEnum
from .base import AgentBase
from .classifier import TaxonomyClassifierAgent
from .configuration import AgentConfiguration, AgentRegistry, get_default_registry
from .coordinator import CoordinatorAgent
from .files import build_agent_input_messages
from .runner import run_async, run_sync
from .search.base import SearchPlanner, SearchToolAgent, SearchWriter
from .search.vector import VectorAgentSearch
from .search.web import WebAgentSearch
from .summarizer import SummarizerAgent
from .translator import TranslatorAgent
from .utils import run_coroutine_agent_sync
from .validator import ValidatorAgent


def __getattr__(name: str) -> Any:
    """Load optional agent integrations only when requested.

    Parameters
    ----------
    name : str
        Requested module attribute.

    Returns
    -------
    Any
        Lazily imported optional agent class.

    Raises
    ------
    AttributeError
        If the requested name is not an optional public export.
    ImportError
        If the required optional dependency is not installed.
    """
    if name != "ExtractorAgent":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_optional_module(
        "openai_sdk_helpers.agent.extractor",
        dependency="langextract",
        extra="extract",
        feature="ExtractorAgent",
    )
    value = module.ExtractorAgent
    globals()[name] = value
    return value


__all__ = [
    "AgentBase",
    "AgentConfiguration",
    "AgentRegistry",
    "get_default_registry",
    "AgentEnum",
    "CoordinatorAgent",
    "run_sync",
    "run_async",
    "run_coroutine_agent_sync",
    "SearchPlanner",
    "SearchToolAgent",
    "SearchWriter",
    "TaxonomyClassifierAgent",
    "SummarizerAgent",
    "TranslatorAgent",
    "ValidatorAgent",
    "ExtractorAgent",
    "VectorAgentSearch",
    "WebAgentSearch",
    "build_agent_input_messages",
]

if find_spec("langextract") is None:
    __all__.remove("ExtractorAgent")

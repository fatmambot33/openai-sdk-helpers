"""Structured output models for OpenAI API interactions.

This module provides Pydantic-based structured output models for defining
schemas, validation, and serialization of AI agent outputs. It includes base
classes, specialized structures for various agent types, and utilities for
generating OpenAI-compatible schema definitions.
"""

from __future__ import annotations

from typing import Any

from .._optional import import_optional_module
from .agent_blueprint import AgentBlueprint
from .base import *
from .classification import (
    ClassificationResult,
    ClassificationSummary,
    ClassificationStep,
    ClassificationStopReason,
    Taxonomy,
    TaxonomyNode,
    format_path_identifier,
    split_path_identifier,
    taxonomy_enum_path,
)
from .plan import *
from .prompt import PromptStructure
from .responses import *
from .summary import *
from .translation import TranslationStructure
from .validation import ValidationResultStructure
from .vector_search import *
from .web_search import *

_EXTRACTION_EXPORTS = {
    "AnnotatedDocumentStructure",
    "AttributeStructure",
    "DocumentStructure",
    "ExampleDataStructure",
    "ExtractionStructure",
}


def __getattr__(name: str) -> Any:
    """Load extraction structures only when the extraction extra is used.

    Parameters
    ----------
    name : str
        Requested module attribute.

    Returns
    -------
    Any
        Lazily imported extraction structure.

    Raises
    ------
    AttributeError
        If the requested name is not a public extraction structure.
    ImportError
        If ``openai-sdk-helpers[extract]`` is not installed.
    """
    if name not in _EXTRACTION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_optional_module(
        "openai_sdk_helpers.structure.extraction",
        dependency="langextract",
        extra="extract",
        feature="Document extraction",
    )
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "StructureBase",
    "SchemaOptions",
    "spec_field",
    "AgentBlueprint",
    "AgentEnum",
    "ClassificationResult",
    "ClassificationSummary",
    "ClassificationStep",
    "ClassificationStopReason",
    "Taxonomy",
    "TaxonomyNode",
    "format_path_identifier",
    "split_path_identifier",
    "taxonomy_enum_path",
    "TaskStructure",
    "PlanStructure",
    "create_plan",
    "execute_task",
    "execute_plan",
    "PromptStructure",
    "SummaryTopic",
    "SummaryStructure",
    "ExtendedSummaryStructure",
    "TranslationStructure",
    "WebSearchStructure",
    "WebSearchPlanStructure",
    "WebSearchItemStructure",
    "WebSearchItemResultStructure",
    "WebSearchReportStructure",
    "VectorSearchReportStructure",
    "VectorSearchItemStructure",
    "VectorSearchItemResultStructure",
    "VectorSearchItemResultsStructure",
    "VectorSearchPlanStructure",
    "VectorSearchStructure",
    "ValidationResultStructure",
    "AnnotatedDocumentStructure",
    "AttributeStructure",
    "DocumentStructure",
    "ExampleDataStructure",
    "ExtractionStructure",
    "assistant_tool_definition",
    "assistant_format",
    "response_tool_definition",
    "response_format",
]

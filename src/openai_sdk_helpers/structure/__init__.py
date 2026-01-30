"""Structured output models for OpenAI API interactions.

This module provides Pydantic-based structured output models for defining
schemas, validation, and serialization of AI agent outputs. It includes base
classes, specialized structures for various agent types, and utilities for
generating OpenAI-compatible schema definitions.

Classes
-------
StructureBase
    Base class for all structured output models with schema generation.
SchemaOptions
    Configuration options for schema generation behavior.
AgentBlueprint
    Structure for designing and planning new agents.
AgentEnum
    Enumeration of available agent types.
TaskStructure
    Representation of a single task in an execution plan.
PlanStructure
    Ordered sequence of tasks for multi-step agent workflows.
PromptStructure
    Structure for prompt design and validation.
SummaryTopic
    Individual topic within a summary.
SummaryStructure
    Basic summary with topic breakdown.
ExtendedSummaryStructure
    Enhanced summary with additional metadata.
TranslationStructure
    Structured translation output.
WebSearchStructure
    Web search results structure.
WebSearchPlanStructure
    Planned web search queries and strategy.
WebSearchItemStructure
    Individual web search item.
WebSearchItemResultStructure
    Result from executing a web search item.
WebSearchReportStructure
    Complete web search report with findings.
VectorSearchStructure
    Vector database search results.
VectorSearchPlanStructure
    Planned vector search queries.
VectorSearchItemStructure
    Individual vector search query.
VectorSearchItemResultStructure
    Result from executing a vector search query.
VectorSearchItemResultsStructure
    Collection of vector search results.
VectorSearchReportStructure
    Complete vector search report.
ValidationResultStructure
    Validation results with pass/fail status.
ExtractionItem
    Extracted item with source span data.
ExtractionResult
    Structured extraction results for a document.

Functions
---------
spec_field
    Create a Pydantic Field with standard documentation formatting.
assistant_tool_definition
    Build function tool definition for Assistant APIs.
assistant_format
    Build response format for Assistant APIs.
response_tool_definition
    Build function tool definition for chat completions.
response_format
    Build response format for chat completions.
"""

from __future__ import annotations

from .agent_blueprint import AgentBlueprint
from .base import *
from .classification import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    Taxonomy,
    TaxonomyNode,
    flatten_taxonomy,
    taxonomy_enum_path,
)
from .extraction import (
    AnnotatedDocumentStructure,
    AttributeStructure,
    DocumentStructure,
    ExampleDataStructure,
    ExtractionStructure,
)
from .plan import *
from .prompt import PromptStructure
from .responses import *
from .summary import *
from .translation import TranslationStructure
from .validation import ValidationResultStructure
from .vector_search import *
from .web_search import *

__all__ = [
    "StructureBase",
    "SchemaOptions",
    "spec_field",
    "AgentBlueprint",
    "AgentEnum",
    "ClassificationResult",
    "ClassificationStep",
    "ClassificationStopReason",
    "Taxonomy",
    "TaxonomyNode",
    "flatten_taxonomy",
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

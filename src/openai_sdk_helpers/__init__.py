"""Shared AI helpers and base structures."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

from .codex import (
    CODEX_PLUGIN_ENTRY_POINT,
    CodexCommand,
    CodexPlugin,
    CodexPluginContext,
    CodexPluginRegistry,
)
from .environment import get_data_path
from .utils.async_utils import run_coroutine_thread_safe, run_coroutine_with_fallback

from .errors import (
    OpenAISDKError,
    ConfigurationError,
    PromptNotFoundError,
    AgentExecutionError,
    VectorStorageError,
    ToolExecutionError,
    ResponseGenerationError,
    InputValidationError,
    AsyncExecutionError,
    ResourceCleanupError,
    ExtractionError,
)

from .utils.validation import (
    validate_choice,
    validate_dict_mapping,
    validate_list_items,
    validate_max_length,
    validate_non_empty_string,
    validate_safe_path,
    validate_url_format,
)
from .structure import (
    StructureBase,
    SchemaOptions,
    PlanStructure,
    TaskStructure,
    WebSearchStructure,
    VectorSearchStructure,
    PromptStructure,
    spec_field,
    SummaryStructure,
    ExtendedSummaryStructure,
    ValidationResultStructure,
    AgentBlueprint,
    create_plan,
    execute_task,
    execute_plan,
)
from .prompt import PromptRenderer
from .settings import OpenAISettings
from .files_api import FilesAPIManager, FilePurpose
from .vector_storage import VectorStorage, VectorStorageFileInfo, VectorStorageFileStats
from .agent import (
    AgentBase,
    AgentConfiguration,
    AgentEnum,
    CoordinatorAgent,
    SummarizerAgent,
    TranslatorAgent,
    ValidatorAgent,
    VectorAgentSearch,
    WebAgentSearch,
)
from .response import (
    ResponseBase,
    ResponseMessage,
    ResponseMessages,
    ResponseToolCall,
    ResponseConfiguration,
    ResponseRegistry,
    get_default_registry,
    attach_vector_store,
    TaxonomyClassifierResponse,
    classify_taxonomy_response,
    TranslatorResponse,
    translate_response,
    open_websocket_connection,
    build_response_create_event,
    send_response_create,
)
from .tools import (
    tool_handler_factory,
    StructureType,
    ToolHandler,
    ToolHandlerRegistration,
    ToolSpec,
    build_tool_definition_list,
)
from .settings import build_openai_settings
from .utils.output_validation import (
    ValidationResult,
    ValidationRule,
    JSONSchemaValidator,
    SemanticValidator,
    LengthValidator,
    OutputValidator,
    validate_output,
)
from .utils.langextract import LangExtractAdapter, build_langextract_adapter

_OPTIONAL_EXPORTS = {
    "ExtractorAgent": ("openai_sdk_helpers.agent", "ExtractorAgent"),
    "AnnotatedDocumentStructure": (
        "openai_sdk_helpers.structure",
        "AnnotatedDocumentStructure",
    ),
    "AttributeStructure": ("openai_sdk_helpers.structure", "AttributeStructure"),
    "DocumentStructure": ("openai_sdk_helpers.structure", "DocumentStructure"),
    "ExampleDataStructure": ("openai_sdk_helpers.structure", "ExampleDataStructure"),
    "ExtractionStructure": ("openai_sdk_helpers.structure", "ExtractionStructure"),
    "DocumentExtractor": ("openai_sdk_helpers.extract", "DocumentExtractor"),
    "EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS": (
        "openai_sdk_helpers.extract",
        "EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS",
    ),
    "EXTRACTOR_CONFIG_GENERATOR": (
        "openai_sdk_helpers.extract",
        "EXTRACTOR_CONFIG_GENERATOR",
    ),
    "PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS": (
        "openai_sdk_helpers.extract",
        "PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS",
    ),
    "generate_document_extractor_config": (
        "openai_sdk_helpers.extract",
        "generate_document_extractor_config",
    ),
    "generate_document_extractor_config_with_agent": (
        "openai_sdk_helpers.extract",
        "generate_document_extractor_config_with_agent",
    ),
    "optimize_extractor_prompt": (
        "openai_sdk_helpers.extract",
        "optimize_extractor_prompt",
    ),
    "optimize_extractor_prompt_with_agent": (
        "openai_sdk_helpers.extract",
        "optimize_extractor_prompt_with_agent",
    ),
}


def __getattr__(name: str) -> Any:
    """Load optional extraction exports only when requested.

    Parameters
    ----------
    name : str
        Requested package attribute.

    Returns
    -------
    Any
        Lazily imported public export.

    Raises
    ------
    AttributeError
        If the requested attribute is not a public optional export.
    ImportError
        If the extraction extra is required but not installed.
    """
    target = _OPTIONAL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "CODEX_PLUGIN_ENTRY_POINT",
    "CodexCommand",
    "CodexPlugin",
    "CodexPluginContext",
    "CodexPluginRegistry",
    "get_data_path",
    "run_coroutine_thread_safe",
    "run_coroutine_with_fallback",
    "OpenAISDKError",
    "ConfigurationError",
    "PromptNotFoundError",
    "AgentExecutionError",
    "VectorStorageError",
    "ToolExecutionError",
    "ResponseGenerationError",
    "InputValidationError",
    "AsyncExecutionError",
    "ResourceCleanupError",
    "ExtractionError",
    "validate_non_empty_string",
    "validate_max_length",
    "validate_url_format",
    "validate_dict_mapping",
    "validate_list_items",
    "validate_choice",
    "validate_safe_path",
    "StructureBase",
    "SchemaOptions",
    "spec_field",
    "PromptRenderer",
    "OpenAISettings",
    "FilesAPIManager",
    "FilePurpose",
    "VectorStorage",
    "VectorStorageFileInfo",
    "VectorStorageFileStats",
    "SummaryStructure",
    "PromptStructure",
    "AgentBlueprint",
    "TaskStructure",
    "PlanStructure",
    "AgentEnum",
    "AgentBase",
    "AgentConfiguration",
    "CoordinatorAgent",
    "ExtractorAgent",
    "SummarizerAgent",
    "TranslatorAgent",
    "ValidatorAgent",
    "VectorAgentSearch",
    "WebAgentSearch",
    "ExtendedSummaryStructure",
    "WebSearchStructure",
    "VectorSearchStructure",
    "ValidationResultStructure",
    "AnnotatedDocumentStructure",
    "AttributeStructure",
    "DocumentStructure",
    "ExampleDataStructure",
    "ExtractionStructure",
    "ResponseBase",
    "ResponseMessage",
    "ResponseMessages",
    "ResponseToolCall",
    "ResponseConfiguration",
    "ResponseRegistry",
    "get_default_registry",
    "attach_vector_store",
    "TaxonomyClassifierResponse",
    "classify_taxonomy_response",
    "TranslatorResponse",
    "translate_response",
    "open_websocket_connection",
    "build_response_create_event",
    "send_response_create",
    "tool_handler_factory",
    "StructureType",
    "ToolHandler",
    "ToolHandlerRegistration",
    "ToolSpec",
    "build_tool_definition_list",
    "build_openai_settings",
    "create_plan",
    "execute_task",
    "execute_plan",
    "ValidationResult",
    "ValidationRule",
    "JSONSchemaValidator",
    "SemanticValidator",
    "LengthValidator",
    "OutputValidator",
    "validate_output",
    "LangExtractAdapter",
    "build_langextract_adapter",
    "DocumentExtractor",
    "EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS",
    "EXTRACTOR_CONFIG_GENERATOR",
    "PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS",
    "generate_document_extractor_config",
    "generate_document_extractor_config_with_agent",
    "optimize_extractor_prompt",
    "optimize_extractor_prompt_with_agent",
]

if find_spec("langextract") is None:
    __all__ = [name for name in __all__ if name not in _OPTIONAL_EXPORTS]

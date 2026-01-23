"""Shared AI helpers and base structures."""

from __future__ import annotations

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
    AnnotatedDocumentStructure,
    AttributeStructure,
    DocumentStructure,
    ExampleDataStructure,
    ExtractionStructure,
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
from .extract import (
    DocumentExtractor,
    EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS,
    EXTRACTOR_CONFIG_GENERATOR,
    PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS,
    generate_document_extractor_config,
    generate_document_extractor_config_with_agent,
    optimize_extractor_prompt,
    optimize_extractor_prompt_with_agent,
)

__all__ = [
    # Environment utilities
    "get_data_path",
    # Async utilities
    "run_coroutine_thread_safe",
    "run_coroutine_with_fallback",
    # Error classes
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
    # Validation
    "validate_non_empty_string",
    "validate_max_length",
    "validate_url_format",
    "validate_dict_mapping",
    "validate_list_items",
    "validate_choice",
    "validate_safe_path",
    # Main structure classes
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
    # Output validation
    "ValidationResult",
    "ValidationRule",
    "JSONSchemaValidator",
    "SemanticValidator",
    "LengthValidator",
    "OutputValidator",
    "validate_output",
    # LangExtract
    "LangExtractAdapter",
    "build_langextract_adapter",
    # Extraction helpers
    "DocumentExtractor",
    "EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS",
    "EXTRACTOR_CONFIG_GENERATOR",
    "PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS",
    "generate_document_extractor_config",
    "generate_document_extractor_config_with_agent",
    "optimize_extractor_prompt",
    "optimize_extractor_prompt_with_agent",
]

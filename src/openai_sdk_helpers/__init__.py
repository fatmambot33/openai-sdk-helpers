"""Shared AI helpers and base structures."""

from __future__ import annotations

from .async_utils import run_coroutine_thread_safe, run_coroutine_with_fallback
from .context_manager import (
    AsyncManagedResource,
    ManagedResource,
    async_context,
    ensure_closed,
    ensure_closed_async,
)
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
)
from .logging_config import LoggerFactory
from .retry import with_exponential_backoff
from .validation import (
    validate_choice,
    validate_dict_mapping,
    validate_list_items,
    validate_max_length,
    validate_non_empty_string,
    validate_safe_path,
    validate_url_format,
)
from .structure import (
    BaseStructure,
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
)
from .prompt import PromptRenderer
from .config import OpenAISettings
from .vector_storage import VectorStorage, VectorStorageFileInfo, VectorStorageFileStats
from .agent import (
    AgentBase,
    AgentConfig,
    AgentEnum,
    CoordinatorAgent,
    SummarizerAgent,
    TranslatorAgent,
    ValidatorAgent,
    VectorSearch,
    WebAgentSearch,
)
from .response import (
    BaseResponse,
    ResponseMessage,
    ResponseMessages,
    ResponseToolCall,
    attach_vector_store,
)

__all__ = [
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
    # Logging
    "LoggerFactory",
    # Retry utilities
    "with_exponential_backoff",
    # Context managers
    "ManagedResource",
    "AsyncManagedResource",
    "ensure_closed",
    "ensure_closed_async",
    "async_context",
    # Validation
    "validate_non_empty_string",
    "validate_max_length",
    "validate_url_format",
    "validate_dict_mapping",
    "validate_list_items",
    "validate_choice",
    "validate_safe_path",
    # Main structure classes
    "BaseStructure",
    "SchemaOptions",
    "spec_field",
    "PromptRenderer",
    "OpenAISettings",
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
    "AgentConfig",
    "CoordinatorAgent",
    "SummarizerAgent",
    "TranslatorAgent",
    "ValidatorAgent",
    "VectorSearch",
    "WebAgentSearch",
    "ExtendedSummaryStructure",
    "WebSearchStructure",
    "VectorSearchStructure",
    "ValidationResultStructure",
    "BaseResponse",
    "ResponseMessage",
    "ResponseMessages",
    "ResponseToolCall",
    "attach_vector_store",
]

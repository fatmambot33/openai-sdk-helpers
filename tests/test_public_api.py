"""Regression tests for the supported package-root API."""

from __future__ import annotations

import openai_sdk_helpers


EXPECTED_PUBLIC_API = (
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
)


def test_package_root_exports_are_explicit_and_stable() -> None:
    """Keep the supported package-root exports intentional and ordered."""
    assert tuple(openai_sdk_helpers.__all__) == EXPECTED_PUBLIC_API
    assert len(openai_sdk_helpers.__all__) == len(set(openai_sdk_helpers.__all__))


def test_every_public_export_is_importable() -> None:
    """Ensure every declared public name exists on the package root."""
    missing = [
        name
        for name in openai_sdk_helpers.__all__
        if not hasattr(openai_sdk_helpers, name)
    ]

    assert missing == []

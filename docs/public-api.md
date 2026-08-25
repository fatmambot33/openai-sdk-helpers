# Supported package-root API

The names below are the intentional public API exposed from
`openai_sdk_helpers`. They can be imported directly from the package root.
Submodule imports remain available, but they are not covered by this package-root
compatibility contract unless listed here.

## Environment and configuration

- `get_data_path`
- `OpenAISettings`
- `build_openai_settings`

## Async helpers

- `run_coroutine_thread_safe`
- `run_coroutine_with_fallback`

## Operation lifecycle

- `OperationContext`
- `OperationEvent`
- `OperationObserver`
- `OperationPhase`
- `OperationUsage`
- `run_observed_async`
- `run_observed_sync`

## Conversation state

- `AgentRunState`
- `ConversationStateMode`
- `LocalMessageStore`
- `ResponseContinuation`

## Errors

- `OpenAISDKError`
- `ConfigurationError`
- `PromptNotFoundError`
- `AgentExecutionError`
- `VectorStorageError`
- `ToolExecutionError`
- `ResponseGenerationError`
- `InputValidationError`
- `AsyncExecutionError`
- `ResourceCleanupError`
- `ExtractionError`

## Input validation

- `validate_non_empty_string`
- `validate_max_length`
- `validate_url_format`
- `validate_dict_mapping`
- `validate_list_items`
- `validate_choice`
- `validate_safe_path`

## Structures and orchestration

- `StructureBase`
- `SchemaOptions`
- `spec_field`
- `SummaryStructure`
- `PromptStructure`
- `AgentBlueprint`
- `TaskStructure`
- `PlanStructure`
- `ExtendedSummaryStructure`
- `WebSearchStructure`
- `VectorSearchStructure`
- `ValidationResultStructure`
- `AnnotatedDocumentStructure`
- `AttributeStructure`
- `DocumentStructure`
- `ExampleDataStructure`
- `ExtractionStructure`
- `create_plan`
- `execute_task`
- `execute_plan`

## Prompts, files, and vector storage

- `PromptRenderer`
- `FilesAPIManager`
- `FilePurpose`
- `VectorStorage`
- `VectorStorageFileInfo`
- `VectorStorageFileStats`

## Agents

- `AgentEnum`
- `AgentBase`
- `AgentConfiguration`
- `CoordinatorAgent`
- `ExtractorAgent`
- `SummarizerAgent`
- `TranslatorAgent`
- `ValidatorAgent`
- `VectorAgentSearch`
- `WebAgentSearch`

## Responses API helpers

- `ResponseBase`
- `ResponseMessage`
- `ResponseMessages`
- `ResponseToolCall`
- `ResponseConfiguration`
- `ResponseRegistry`
- `get_default_registry`
- `attach_vector_store`
- `TaxonomyClassifierResponse`
- `classify_taxonomy_response`
- `TranslatorResponse`
- `translate_response`
- `open_websocket_connection`
- `build_response_create_event`
- `send_response_create`

## Tool helpers

- `tool_handler_factory`
- `StructureType`
- `ToolHandler`
- `ToolHandlerRegistration`
- `ToolSpec`
- `build_tool_definition_list`

## Output validation

- `ValidationResult`
- `ValidationRule`
- `JSONSchemaValidator`
- `SemanticValidator`
- `LengthValidator`
- `OutputValidator`
- `validate_output`

## LangExtract helpers

- `LangExtractAdapter`
- `build_langextract_adapter`

## Extraction helpers

- `DocumentExtractor`
- `EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS`
- `EXTRACTOR_CONFIG_GENERATOR`
- `PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS`
- `generate_document_extractor_config`
- `generate_document_extractor_config_with_agent`
- `optimize_extractor_prompt`
- `optimize_extractor_prompt_with_agent`

## Focused MCP namespace

MCP remains intentionally absent from the package root. The supported preview
surface is exported explicitly from `openai_sdk_helpers.mcp`:

- `HostedMCPConfig`
- `MCPServerProtocol`
- `MCPTransport`
- `ManagedMCPServer`
- `StreamableHTTPMCPConfig`
- `build_hosted_mcp_tool`
- `build_streamable_http_server`

This focused namespace preserves the underlying official SDK objects and does
not imply connection, discovery, trust, or tool execution when imported.

## Compatibility policy

The package-root export list is defined by `openai_sdk_helpers.__all__` and is
covered by regression tests. Removing or renaming one of these names is a
breaking API change and must follow the project's deprecation and semantic
versioning policy. Focused namespaces with their own explicit `__all__` contract,
including `openai_sdk_helpers.mcp`, follow the same deliberate review policy for
removals or renames.

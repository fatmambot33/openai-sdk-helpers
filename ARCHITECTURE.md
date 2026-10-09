# Architecture & Design Decisions

This document outlines key architectural decisions and patterns used in `openai-sdk-helpers`.

## Core Design Principles

### 1. Dual SDK Support

**Decision**: Support both `openai` and `openai-agents` SDKs simultaneously.

**Rationale**: 
- `openai-agents` provides high-level abstractions for agent workflows
- `openai` offers fine-grained control for custom response handling
- Different use cases require different levels of abstraction

**Implementation**:
- `agent/` module → uses `openai-agents`
- `response/` module → uses `openai`
- Shared utilities work with both

### 2. Type Safety First

**Decision**: Full type hints with `py.typed` marker for external type checkers.

**Rationale**:
- Catches errors at development time
- Improves IDE autocomplete and documentation
- Enables refactoring with confidence

**Implementation**:
- All public APIs have type hints
- Pydantic models for complex data structures
- `py.typed` file ships with package

### 3. Composable Primitives

**Decision**: Provide low-level building blocks instead of opinionated frameworks.

**Rationale**:
- Users have diverse requirements
- Flexibility over convention
- Easy to integrate into existing codebases

**Implementation**:
- Each module is independently usable
- Minimal coupling between components
- Clear separation of concerns

## Module Architecture

### Agent Module (`agent/`)

High-level abstractions built on `openai-agents` SDK.

**Components**:
- `base.py` - Base agent class with sync/async support
- `coordinator.py` - Multi-agent coordination
- `search/` - Vector and web search workflows
- Text agents (summarizer, translator, validator)

**Design Pattern**: Template Method pattern with hooks for customization.

### Response Module (`response/`)

Direct API control with `openai` SDK.

**Components**:
- `base.py` - Response handling primitives
- `configuration.py` - Configuration and registry
- `tool_call.py` - Tool execution framework

**Design Pattern**: Builder pattern for constructing API calls.

### Decisions Module (`decisions.py`)

Capability-gated access to the official Decisions API for predicate, choice, and
score requests.

**Design constraints**:
- preserve official SDK question and answer types;
- preserve raw `client.decisions` access and native `Decision` results;
- provide matching sync/async facades;
- keep feature availability structural so older package-supported SDKs remain
  import-safe;
- avoid building a routing framework or policy engine on top of the API.

### Infrastructure

#### Logging (`logging_config.py`)

**Decision**: Centralize logger creation and structured formatting.

**Patterns**:
- Structured logging with correlation IDs
- Cached logger factory to avoid duplicate configuration
- Configurable log levels

#### Retry (`retry.py`)

**Decision**: Handle transient failures with configurable backoff.

**Rationale**:
- Retry handles transient failures (rate limits, network issues)
- Aligns with provider guidance on rate limit recovery

**Patterns**:
- Exponential backoff with jitter
- Decorator pattern for easy application

#### Output Validation (`output_validation.py`)

**Decision**: Composable validation rules instead of monolithic validator.

**Rationale**:
- Different outputs need different validations
- Rules can be combined flexibly
- Easy to add custom validators

**Patterns**:
- Strategy pattern for validation rules
- Composite pattern for combining validators
- Builder pattern for convenience functions

## Error Handling Strategy

### Exception Hierarchy

```
OpenAISDKError (base)
├── ConfigurationError
├── PromptNotFoundError
├── AgentExecutionError
├── VectorStorageError
├── ToolExecutionError
├── ResponseGenerationError
├── InputValidationError
├── AsyncExecutionError
└── ResourceCleanupError
```

### Error Recovery

1. **Transient Errors** → Retry with exponential backoff
2. **Persistent Errors** → Surface failure after retries are exhausted
3. **Validation Errors** → Early failure with clear messages
4. **Resource Errors** → Cleanup with context managers

## Performance Considerations

### Async-First Design

- All network operations support async/await
- Sync wrappers provided for convenience
- Semaphores limit concurrent operations

### Caching

- Template compilation cached (LRU)
- Logger instances cached per name
- Configuration registry is singleton

### Rate Limiting

- Batch processor respects RPM limits
- Exponential backoff reduces API load

## Testing Strategy

### Unit Tests

- Each module has dedicated test file
- Mocked external dependencies
- 70%+ code coverage requirement

### Integration Tests

- End-to-end workflows tested
- Real API calls in CI (with rate limiting)
- Smoke tests for critical paths

### Type Checking

- `pyright` in strict mode
- All public APIs fully typed
- CI blocks on type errors

## Deprecation Policy

### Process

1. Mark feature with `@deprecated` decorator
2. Document alternative in deprecation message
3. Keep deprecated feature for 1 minor version
4. Remove in next major version

### Example

```python
from openai_sdk_helpers import deprecated

@deprecated("2.0.0", alternative="new_function")
def old_function():
    pass
```

## Version Compatibility

See [COMPATIBILITY.md](COMPATIBILITY.md) for SDK version matrix.

### Semantic Versioning

- **Major**: Breaking changes to public API
- **Minor**: New features, deprecated features
- **Patch**: Bug fixes, documentation

## Future Considerations

### Potential Additions

1. **OpenTelemetry Integration** - For production observability
2. **Prompt Template Versioning** - Track template changes
3. **A/B Testing Framework** - Compare prompt variants
4. **Hybrid Vector Search** - Combine embedding + text search
5. **Multi-LLM Support** - Abstract beyond OpenAI

### Non-Goals

- **Training/Fine-tuning** - Out of scope
- **Model Serving** - Use dedicated platforms
- **UI Components** - Keep library-focused
- **Data Pipelines** - Use specialized tools

## Contributing

See [AGENTS.md](AGENTS.md) for code style and testing requirements.

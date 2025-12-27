# Repository Review

This document captures improvement opportunities observed during a quick audit of the current codebase. Items are organized by impact area to help prioritize follow-up work.

## TODO
- [ ] Fail fast when prompts are missing in `BaseAgent`.
- [ ] Make vector search concurrency configurable in `VectorSearch`/`VectorSearchTool`.
- [ ] Isolate failures across parallel vector searches to allow partial results.
- [ ] Inject vector storage dependencies into `VectorSearchTool` for easier testing and extensibility.
- [ ] Add validation for missing OpenAI configuration in `OpenAISettings`/`BaseAgent` setup.
- [ ] Add tracing spans for individual stages in the vector search workflow.

## Agent ergonomics
- **Fail fast when prompts are missing.** `BaseAgent` silently renders an empty template if a prompt file is absent, which can hide configuration mistakes. Adding an explicit warning or raising a descriptive exception when the derived template path does not exist would make misconfigurations obvious and simplify debugging for new agents.【F:src/openai_sdk_helpers/agent/base.py†L77-L105】
- **Make vector search concurrency configurable.** The vector search flow pins concurrency to the module-level `MAX_CONCURRENT_SEARCHES` constant. Allowing the limit to be passed into `VectorSearch` or `VectorSearchTool` (with the current default preserved) would let consumers tune throughput for different hardware or vector store backends.【F:src/openai_sdk_helpers/agent/vector_search.py†L24-L182】
- **Isolate failures in parallel searches.** The `asyncio.gather` call in `VectorSearchTool.run_agent` will bubble up the first exception and cancel sibling tasks. Wrapping individual searches with error handling (or using `return_exceptions=True` plus result filtering) would prevent a single vector store failure from aborting the entire batch and enable partial results with clearer reporting.【F:src/openai_sdk_helpers/agent/vector_search.py†L152-L210】

## Extensibility and testing
- **Inject vector storage dependencies.** `VectorSearchTool` constructs `VectorStorage` internally based on a store name. Accepting a storage instance or factory in the constructor would make it easier to plug in alternative stores and to unit test search logic without touching the filesystem or networked vector services.【F:src/openai_sdk_helpers/agent/vector_search.py†L113-L210】
- **Surface OpenAI configuration issues earlier.** `BaseAgent` raises when no model is provided, but `OpenAISettings` currently allows all fields to remain `None`. Adding a `validate` helper (or raising in `from_env` when credentials are absent) would give quicker feedback when environment variables are missing before any agent is instantiated.【F:src/openai_sdk_helpers/config.py†L12-L75】【F:src/openai_sdk_helpers/agent/base.py†L77-L105】

## Observability
- **Add tracing context to search stages.** The vector search workflow already wraps the overall run in a trace but does not annotate individual stages (plan, search, write). Emitting spans around each stage would make it easier to correlate agent prompt outputs and vector store performance in distributed traces.【F:src/openai_sdk_helpers/agent/vector_search.py†L137-L210】

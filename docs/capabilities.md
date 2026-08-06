# Capability matrix

This document is the canonical inventory of supported package capabilities.
README content should summarize and link here rather than duplicate the matrix.

Maturity meanings:

- **Stable** — intentionally public and covered by compatibility policy.
- **Supported** — production-usable, but still expected to evolve additively.
- **Preview** — available for experimentation; compatibility may be narrower.
- **Planned** — roadmap only and not part of the current package surface.

## Current capabilities

| Capability | Primary surface | SDK relationship | Maturity | Installation | Execution | Escape hatch |
| --- | --- | --- | --- | --- | --- | --- |
| Settings and client creation | `openai_sdk_helpers.settings` | Configures the official OpenAI Python SDK | Stable | Core | Sync construction | Returns configured official SDK clients and accepts extra client keyword arguments |
| Shared operation context | `openai_sdk_helpers.runtime` | Vendor-neutral lifecycle metadata that complements, but does not replace, official SDK tracing | Preview | Core | Sync and async observers | Original results and exceptions remain unchanged; Agents SDK tracing remains directly configurable |
| Explicit conversation state | `openai_sdk_helpers.state` and Agents runners | Validates official Agents session and server-continuation choices; provides explicit local `ResponseMessages` storage | Preview | Core | Local validation plus matching sync/async runner forwarding | Underlying session objects, response IDs, conversation IDs, request kwargs, and message collections remain accessible |
| Retrieval lifecycle | `openai_sdk_helpers.retrieval` | Wraps injected official Files and Vector Stores resources without owning credentials or cleanup | Preview | Core | Matching sync/async clients; explicit SDK polling | Normalized outcomes preserve raw SDK resources and exceptions; `sdk_client` exposes the injected official client |
| Direct search and File Search | `openai_sdk_helpers.retrieval` | Wraps official vector-store search and builds official Responses/Agents File Search configuration | Preview | Core | Matching sync/async direct search plus local hosted-tool adapters | Raw search pages, result items, tool calls, citations, filters, request mappings, and official Agents tools remain accessible |
| Responses workflows | `openai_sdk_helpers.response` | Thin orchestration over the official Responses API | Supported | Core | Sync and async paths; websocket helpers are async/stream-oriented where appropriate | Callers retain SDK configuration, response identifiers, raw events, and result objects |
| Agents workflows | `openai_sdk_helpers.agent` | Composes the official OpenAI Agents SDK | Supported | Core | Sync and async runners | Callers retain underlying Agents SDK objects, tools, sessions, and results |
| Typed structures | `openai_sdk_helpers.structure` | Pydantic schemas for SDK inputs and outputs | Stable | Core; extraction structures require `extract` | Local | Pydantic models and generated schemas remain directly accessible |
| Prompt rendering | `openai_sdk_helpers.prompt` | SDK-independent Jinja rendering | Stable | Core | Local | Callers control template directories and rendered strings |
| Tool contracts and handlers | `openai_sdk_helpers.tools` | Reusable definitions for Responses and Agents integrations | Supported | Core | Sync and async handlers where declared | Raw tool definitions and handler exceptions remain accessible |
| Codex plugin surface | `openai_sdk_helpers.codex` and `openai_sdk_helpers.codex_cli` | Package entry-point plugin contract | Stable for 0.8 | Core | Sync and async commands; startup and shutdown lifecycle | Registry, plugin metadata, discovery reports, and original command handlers remain accessible |
| Files API helpers | `openai_sdk_helpers.files_api` | Legacy thin helpers over official Files resources | Supported; compatibility adapters planned for 0.9 | Core | Sync | Underlying OpenAI client and file resources remain accessible |
| Vector-store helpers | `openai_sdk_helpers.vector_storage` and response vector-store helpers | Legacy helpers over official Vector Stores and File Search resources | Supported; compatibility adapters planned for 0.9 | Core | Primarily sync in the current public surface | Underlying client, store identifiers, and SDK resources remain accessible |
| Responses websocket helpers | `openai_sdk_helpers.response.websocket` | Wraps the official Responses websocket connection | Preview | Core | Streaming / connection-oriented | Raw connection and events remain accessible |
| Output validation | `openai_sdk_helpers.utils.output_validation` | SDK-independent validation adapters | Stable | Core | Local | Individual validators and original values remain accessible |
| Document extraction | `openai_sdk_helpers.extract`, `ExtractorAgent`, extraction structures | Optional LangExtract integration plus Agents helpers | Supported | `openai-sdk-helpers[extract]` | Local and agent-driven paths | LangExtract objects, extraction models, and agent results remain accessible |
| Streamlit application helpers | `openai_sdk_helpers.streamlit_app` | Optional UI composition layer | Supported | `openai-sdk-helpers[ui]` | Interactive | Callers own Streamlit configuration and page/application composition |
| CLI inspection | `openai-helpers`, `openai-helpers-credentials` | Local package and configuration inspection | Supported | Core | Local command line | Commands expose registry and plugin data without hidden API calls |

## Planned capabilities

Planned items are not installed, exported, or implied by the current package.
They must pass the feature acceptance test in `PRODUCT.md` before implementation.

| Capability | Target | Roadmap status | Constraint |
| --- | --- | --- | --- |
| MCP integration | `0.10.0` | Issues #143–#144 | Optional extra; explicit filtering, approvals, lifecycle, and failure isolation |
| Realtime integration | `0.11.0` | Issues #145–#146 | Server-side lifecycle and event helpers only; no browser or audio-device application layer |

Images and audio generation are not committed roadmap surfaces. They should be
added only after repeated workflows demonstrate that a package-level helper is
smaller and clearer than direct official SDK usage.

## Choosing a surface

Use **Responses** when the application needs direct control of request inputs,
response identifiers, message history, tool dispatch, or raw events.

Use **Agents** when the application benefits from the official Agents SDK's
agent loop, handoffs, tools, sessions, guardrails, and tracing.

Use **Codex plugins** when a separately packaged capability should register
commands through deterministic entry-point discovery without modifying the core
package.

Use direct official SDK calls when a helper would only rename parameters or hide
resource ownership. The package is not intended to replace either official SDK.

## Documentation ownership

A pull request that changes a public capability must update all applicable
canonical documents:

1. this capability matrix for surface, maturity, installation, or execution changes;
2. `docs/public-api.md` for public import changes;
3. `docs/installation.md` for dependency-profile changes;
4. the relevant focused guide for behavior and examples;
5. `CHANGELOG.md` for user-visible changes;
6. `README.md` only when the top-level summary or primary navigation changes.

The README must not grow a second capability inventory. Detailed behavior belongs
in focused guides, and internal links are validated in CI.

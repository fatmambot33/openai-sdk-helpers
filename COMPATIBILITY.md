# SDK Compatibility Matrix

This document tracks tested and supported versions of the OpenAI SDKs.

## Supported SDK Versions

| openai-sdk-helpers | openai | openai-agents | Python | Status |
|---|---:|---:|---:|---|
| 0.7.x | >=2.45.0,<3 | >=0.18.1,<1 | 3.10-3.13 | Active |

## SDK Version Details

### OpenAI Python SDK (`openai`)

The `openai` package powers direct Responses API interactions, structured
outputs, tools, streaming, files, and vector stores.

- **Minimum supported version:** 2.45.0
- **Supported major version:** 2.x
- **Primary API:** Responses API

The baseline includes the current 2.x response schemas and the newer client
transport behavior available in the July 2026 SDK generation.

### OpenAI Agents SDK (`openai-agents`)

The `openai-agents` package powers higher-level agent workflows.

- **Minimum supported version:** 0.18.1
- **Supported pre-1.0 range:** 0.18.1 and later, below 1.0

Version 0.18.0 is intentionally excluded because its default usage model can
fail during `RunContextWrapper` construction with supported Pydantic releases.
The Agents SDK is pre-1.0 and may introduce public API changes in minor
releases. Application code should set models explicitly when reproducible
behavior matters instead of relying on SDK defaults.

## Version Constraints

Current constraints in `pyproject.toml`:

```toml
dependencies = [
    "openai>=2.45.0,<3.0.0",
    "openai-agents>=0.18.1,<1.0.0",
]
```

## Testing Strategy

Compatibility CI runs two dependency modes:

1. **Minimum:** installs the declared minimum OpenAI SDK versions.
2. **Latest:** installs the newest versions allowed by the declared upper bounds.

The normal test matrix covers Python 3.10 through 3.13. Tests must remain
network-free unless explicitly marked as integration tests.

## Release-note Alignment

The compatibility baseline accounts for:

- OpenAI Python SDK 2.x Responses API evolution and WebSocket transport work.
- OpenAI Agents SDK Responses transport improvements.
- Agents SDK Realtime default model updates.
- The OpenAI SDK ecosystem minimum runtime of Python 3.10 or later.

Features from upstream SDKs are exposed through their native typed interfaces
unless a reusable helper abstraction is justified by multiple project use
cases. This avoids duplicating fast-moving SDK APIs.

## Known Compatibility Notes

- Explicitly configure the model; upstream SDK defaults can change.
- Pin a narrower `openai-agents` range in applications that require strict
  behavioral reproducibility.
- Use the minimum dependency CI job before raising either lower bound.

## Reporting Issues

When reporting a compatibility problem, include:

1. Python version.
2. `openai-sdk-helpers` version.
3. `openai` and `openai-agents` versions.
4. A minimal reproduction that does not require credentials where possible.

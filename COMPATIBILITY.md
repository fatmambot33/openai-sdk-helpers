# SDK Compatibility Matrix

This document tracks tested and supported versions of the OpenAI SDKs.

## Supported SDK Versions

| openai-sdk-helpers | openai | openai-agents | Python | Status |
|---|---:|---:|---:|---|
| 0.9.x–0.10.x | >=2.45.0,<4 | >=0.18.1,<1 | 3.10-3.13 | Active |

## SDK Version Details

### OpenAI Python SDK (`openai`)

The `openai` package powers direct Responses API interactions, structured
outputs, tools, streaming, files, vector stores, and the managed Agents API when
the installed SDK exposes it.

- **Minimum package-wide supported version:** 2.45.0
- **Supported range:** 2.45.0 and later, below 4.0
- **Managed Agents API feature minimum:** 3.13.0

Existing package surfaces remain supported on the package-wide minimum. The
`openai_sdk_helpers.managed_agents` module is safe to import on earlier supported
versions, but constructing its facade requires a client exposing
`beta.agents.sessions`. If that capability is absent, the helper raises
`ManagedAgentsUnavailableError` before any network request with the installed
version and feature minimum in its context.

This feature-specific requirement avoids raising the dependency floor for users
that do not need the managed Agents API.

### OpenAI Agents SDK (`openai-agents`)

The `openai-agents` package powers application-run higher-level agent workflows.
It is separate from the managed Agents API exposed by the `openai` package.

- **Minimum supported version:** 0.18.1
- **Supported pre-1.0 range:** 0.18.1 and later, below 1.0

Version 0.18.0 is intentionally excluded because its default usage model can
fail during `RunContextWrapper` construction with supported Pydantic releases.
The Agents SDK is pre-1.0 and may introduce public API changes in minor
releases. Application code should set models explicitly when reproducible
behavior matters instead of relying on SDK defaults.

## Version Constraints

Current package-wide constraints in `pyproject.toml`:

```toml
dependencies = [
    "openai>=2.45.0,<4.0.0",
    "openai-agents>=0.18.1,<1.0.0",
]
```

Managed Agents applications additionally need:

```text
openai>=3.13.0,<4.0.0
```

This is a feature capability requirement, not a package-wide dependency change.

## Testing Strategy

Compatibility validation covers:

1. **Package minimum:** existing helpers remain importable and functional with
   the declared minimum OpenAI SDK dependency set.
2. **Current supported SDK:** managed Agents helpers are tested against clients
   exposing the official `beta.agents.sessions` resource.
3. **Capability failure:** an incompatible client fails before network execution
   with an actionable feature/version error.
4. **Python matrix:** Python 3.10 through 3.13 remains supported.

Tests remain network-free unless explicitly marked as integration tests. The
installed-wheel smoke suite constructs the managed Agents facade without making
an API request.

## Release-note Alignment

The compatibility baseline accounts for:

- OpenAI Python SDK Responses API and WebSocket evolution across supported 2.x
  and 3.x releases.
- OpenAI Python SDK 3.13.0 introducing the managed Agents API.
- OpenAI Agents SDK Responses transport improvements.
- Agents SDK Realtime default model updates.
- The OpenAI SDK ecosystem minimum runtime of Python 3.10 or later.

Features from upstream SDKs are exposed through their native typed interfaces
unless a reusable helper abstraction is justified by repeated SDK plumbing. This
avoids duplicating fast-moving SDK APIs.

## Known Compatibility Notes

- Explicitly configure the model; upstream SDK defaults can change.
- Pin a narrower `openai-agents` range in applications that require strict
  behavioral reproducibility.
- A package-compatible `openai` version below 3.13.0 does not provide the managed
  Agents API; use `managed_agents_available()` before constructing that facade
  when applications support multiple SDK generations.
- Use minimum-dependency CI before raising either package-wide lower bound.

## Reporting Issues

When reporting a compatibility problem, include:

1. Python version.
2. `openai-sdk-helpers` version.
3. `openai` and `openai-agents` versions.
4. A minimal reproduction that does not require credentials where possible.

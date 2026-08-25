# Installation profiles

`openai-sdk-helpers` keeps optional integrations out of the base installation.
Install only the capabilities used by the application.

## Core

```bash
pip install openai-sdk-helpers
```

The core profile includes the OpenAI Python SDK, the OpenAI Agents SDK,
Pydantic, Jinja, settings helpers, Responses helpers, Agents helpers, vector
storage, tools, validation, and the Codex plugin surface. It does not install
LangExtract or Streamlit. The Agents SDK may bring a compatible MCP runtime
transitively, but the package does not treat that transitive dependency as the
stable installation contract for its MCP adapter surface.

## Document extraction

```bash
pip install "openai-sdk-helpers[extract]"
```

The `extract` profile installs LangExtract and enables:

- `DocumentExtractor`
- `ExtractorAgent`
- extraction structures such as `DocumentStructure`
- extraction prompt generation and optimization helpers

Existing package-root imports remain supported. Accessing an extraction export
without the extra raises an `ImportError` containing the installation command.

## Streamlit UI

```bash
pip install "openai-sdk-helpers[ui]"
```

The `ui` profile installs Streamlit and enables the configuration-driven
`openai_sdk_helpers.streamlit_app` surface.

## MCP transports

```bash
pip install "openai-sdk-helpers[mcp]"
```

The `mcp` profile is the stable installation contract for the focused
`openai_sdk_helpers.mcp` namespace. It enables typed hosted MCP configuration
and explicit Agents SDK Streamable HTTP server lifecycle helpers. Importing the
namespace performs no server discovery, connection, or tool execution.

MCP transport builders retain the official SDK tool/server objects and fail with
this exact installation command when the required integration cannot be loaded.
Filtering, approval policy composition, caching, retry policy, and failure
isolation are a separate 0.10 layer and are not implied by installing the extra.

## All optional capabilities

```bash
pip install "openai-sdk-helpers[all]"
```

Use this profile for environments that need document extraction, Streamlit UI,
and MCP transport integration.

## Development

```bash
pip install -e ".[dev]"
```

The development profile includes test, formatting, type-checking, extraction,
and UI dependencies so the complete repository test suite can run locally.
Clean-install CI separately validates `core`, `extract`, `ui`, `mcp`, and `all`
to prevent optional capability contracts from drifting.

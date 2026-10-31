# Installation profiles

`openai-sdk-helpers` keeps optional integrations out of the base installation.
Install only the capabilities used by the application.

## Core

```bash
pip install openai-sdk-helpers
```

The core profile includes the OpenAI Python SDK, the OpenAI Agents SDK,
Pydantic, Jinja, settings helpers, Responses helpers, Agents helpers, managed
Agents capability detection and session helpers, Decisions capability detection
and helpers, Realtime capability detection and server-side helpers, vector
storage, tools, validation, and the Codex plugin surface.
It does not install LangExtract or Streamlit.

The package-wide SDK requirements are `openai>=3.41.0,<4.0.0` and
`openai-agents>=0.24.0,<1.0.0` for every installation profile, including
Managed Agents, Decisions, and Realtime.

The Managed Agents facade checks `client.beta.agents.sessions`, and the
Decisions facade checks `client.decisions.create`. The Realtime facade checks
`client.realtime.connect` and `client.realtime.client_secrets.create`. All
three fail before sending a network request if an injected client lacks the
required capability.

Opening an official Realtime WebSocket connection additionally requires the
OpenAI SDK's optional transport dependency:

```bash
pip install "openai[realtime]>=3.41.0,<4.0.0"
```

Client-secret session creation itself uses the base OpenAI SDK; applications
must handle the returned secret as a credential.

These SDKs are dependencies of the base distribution; no separate
feature-specific installation is required.

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

## All optional capabilities

```bash
pip install "openai-sdk-helpers[all]"
```

Use this profile for environments that need both document extraction and the
Streamlit UI.

## Development

```bash
pip install -e ".[dev]"
```

The development profile includes test, formatting, type-checking, extraction,
and UI dependencies so the complete repository test suite can run locally.
Clean-install CI separately validates `core`, `extract`, `ui`, and `all` to
prevent optional dependencies from leaking back into the base package.

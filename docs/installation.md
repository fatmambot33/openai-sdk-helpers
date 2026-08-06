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
LangExtract or Streamlit.

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

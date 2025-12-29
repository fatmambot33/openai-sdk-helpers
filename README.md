<div align="center">

# openai-sdk-helpers

Shared primitives for composing OpenAI agent workflows: structures, response
handling, prompt rendering, and reusable agent factories.

</div>

## Overview

`openai-sdk-helpers` packages the common building blocks required to assemble agent-driven
applications. The library intentionally focuses on reusable primitives—data
structures, configuration helpers, and orchestration utilities—while leaving
application-specific prompts and tools to the consuming project.

### Features

- **Agent wrappers** for OpenAI Agents SDK with synchronous and asynchronous
  entry points.
- **Prompt rendering** powered by Jinja for dynamic agent instructions.
- **Typed structures** for prompts, responses, and search workflows to ensure
  predictable inputs and outputs.
- **Vector and web search flows** that coordinate planning, execution, and
  reporting.
- **Reusable text agents** for summarization and translation tasks.

## Installation

Install the package directly from PyPI to reuse it across projects:

```bash
pip install openai-sdk-helpers
```

For local development, install with editable sources and the optional dev
dependencies:

```bash
pip install -e .
pip install -e . --group dev
```

## Quickstart

Create a basic vector search workflow by wiring your own prompt templates and
preferred model configuration:

```python
from pathlib import Path

from openai_sdk_helpers.agent.vector_search import VectorSearch


prompts = Path("./prompts")
vector_search = VectorSearch(prompt_dir=prompts, default_model="gpt-4o-mini")

report = vector_search.run_agent_sync("Explain quantum entanglement for beginners")
print(report.report)
```

### Text utilities

Use the built-in text helpers when you need lightweight single-step agents.

```python
from openai_sdk_helpers.agent import (
    SummarizerAgent,
    TranslatorAgent,
    ValidatorAgent,
)


summarizer = SummarizerAgent(default_model="gpt-4o-mini")
translator = TranslatorAgent(default_model="gpt-4o-mini")
validator = ValidatorAgent(default_model="gpt-4o-mini")

summary = summarizer.run_sync("Long-form content to condense")
translation = translator.run_sync("Bonjour", target_language="English")
guardrails = validator.run_sync(
    "Share meeting notes with names removed", agent_output=summary.text
)
```

You can plug in your own prompt templates by placing matching `.jinja` files in
the provided `prompt_dir` and naming them after the agent (for example,
`vector_planner.jinja`).

The library also ships default prompts for the bundled text agents under
`src/openai_sdk_helpers/prompt`, which are used automatically when no custom
prompt directory is provided.

### Centralized OpenAI configuration

`openai-sdk-helpers` ships with a lightweight `OpenAISettings` helper so projects can share
consistent authentication, routing, and model defaults when using the OpenAI
SDK:

```python
from openai_sdk_helpers import OpenAISettings


# Load from environment variables or a local .env file
settings = OpenAISettings.from_env()
client = settings.create_client()

# Reuse the default model across agents
vector_search = VectorSearch(
    prompt_dir=prompts, default_model=settings.default_model or "gpt-4o-mini"
)
```

The helper reads `OPENAI_API_KEY`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`,
`OPENAI_BASE_URL`, and `OPENAI_MODEL` by default but supports overrides for
custom deployments.

## Development

The repository is configured for a lightweight Python development workflow.
Before opening a pull request, format and validate your changes locally:

```bash
# Style and formatting
pydocstyle src
black --check .

# Static type checking
pyright src

# Unit tests with coverage
pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
```

## Project Structure

- `src/openai_sdk_helpers/agent`: Agent factories, orchestration helpers, and search
  workflows.
- `src/openai_sdk_helpers/prompt`: Prompt rendering utilities backed by Jinja.
- `src/openai_sdk_helpers/response`: Response parsing and transformation helpers.
- `src/openai_sdk_helpers/structure`: Typed data structures shared across workflows.
- `src/openai_sdk_helpers/vector_storage`: Minimal vector store abstraction.
- `tests/`: Unit tests covering core modules and structures.

## Key modules

The package centers around a handful of cohesive building blocks:

- `openai_sdk_helpers.agent.project_manager.ProjectManager` coordinates prompt
  creation, plan building, task execution, and summarization while persisting
  intermediate artifacts to disk.
- `openai_sdk_helpers.agent.vector_search.VectorSearch` bundles the planners,
  executors, and summarizers required to run a multi-turn vector search flow
  from a single entry point.
- `openai_sdk_helpers.agent.summarizer.SummarizerAgent`,
  `agent.translator.TranslatorAgent`, and `agent.validator.ValidatorAgent`
  expose streamlined text-processing utilities that reuse shared prompt
  templates.
- `openai_sdk_helpers.response` contains the response runners and helpers used
  to normalize outputs from agents, including streaming responses.
- `openai_sdk_helpers.utils` holds JSON serialization helpers, logging
  utilities, and common validation helpers used across modules.

## Contributing

Contributions are welcome! Please accompany functional changes with relevant
tests and ensure all quality gates pass. Follow the NumPy-style docstring
conventions outlined in `AGENTS.md` to keep the codebase consistent.


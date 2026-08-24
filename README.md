<div align="center">

# openai-sdk-helpers

[![PyPI version](https://img.shields.io/pypi/v/openai-sdk-helpers.svg)](https://pypi.org/project/openai-sdk-helpers/)
[![Python versions](https://img.shields.io/pypi/pyversions/openai-sdk-helpers.svg)](https://pypi.org/project/openai-sdk-helpers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Small, typed, SDK-first primitives for composing OpenAI Responses, Agents,
files, vector stores, tools, and Codex plugins.

[Install](#installation) · [Choose a surface](#choose-a-surface) ·
[Capabilities](docs/capabilities.md) · [Security](SECURITY.md) ·
[Roadmap](ROADMAP.md)

</div>

## Product contract

`openai-sdk-helpers` complements the official `openai` and `openai-agents`
Python packages. It provides reusable production primitives without replacing
the SDKs, hiding API calls, owning application prompts, or becoming a universal
agent framework.

Every public helper should be:

- typed and predictable;
- thin enough that official SDK concepts remain recognizable;
- explicit about network calls, resource ownership, and cleanup;
- usable without unrelated optional integrations;
- tested without requiring paid API calls in pull-request CI;
- backward-compatible or accompanied by deliberate migration guidance.

See [PRODUCT.md](PRODUCT.md) for the full product vision and feature acceptance
test.

## Installation

The core package installs both official OpenAI Python SDKs and no UI or
LangExtract dependency:

```bash
pip install openai-sdk-helpers
```

Optional profiles:

```bash
pip install "openai-sdk-helpers[extract]"  # LangExtract-backed extraction
pip install "openai-sdk-helpers[ui]"       # Streamlit application helpers
pip install "openai-sdk-helpers[all]"      # All current optional capabilities
```

Python 3.10–3.13 is validated in CI. The distribution includes `py.typed`.
See [docs/installation.md](docs/installation.md) for profile details and
missing-extra behavior.

## Choose a surface

### Responses

Use `openai_sdk_helpers.response` when the application needs direct control of
Responses API inputs, response identifiers, message history, custom tool
handlers, files, or raw streaming events.

```python
from openai_sdk_helpers import OpenAISettings
from openai_sdk_helpers.response import ResponseBase

settings = OpenAISettings.from_env()

with ResponseBase(
    name="reviewer",
    instructions="Review the supplied code.",
    tools=None,
    output_structure=None,
    tool_handlers={},
    openai_settings=settings,
) as response:
    result = response.run_sync("Review: def add(a, b): return a + b")
    print(result)
```

### Agents

Use `openai_sdk_helpers.agent` when the application benefits from the official
Agents SDK loop, tools, sessions, guardrails, handoffs, or tracing.

```python
from openai_sdk_helpers.agent import SummarizerAgent

agent = SummarizerAgent(default_model="your-model")
result = agent.run_sync("Summarize this text in one sentence.")
print(result.text)
```

### Codex plugins

Use `openai_sdk_helpers.codex` when a separately packaged capability should
register typed commands through deterministic entry-point discovery.

```bash
openai-helpers codex plugins
openai-helpers codex commands
```

See [docs/codex-plugins.md](docs/codex-plugins.md) for the plugin protocol,
lifecycle, discovery, compatibility, and packaging guide.

### Direct SDK calls

Use the official SDK directly when a package helper would only rename
parameters or obscure the underlying client, resource, response, event, or
exception. Access to underlying SDK objects is part of this project's escape-
hatch policy.

The canonical inventory of shipped and planned surfaces is
[docs/capabilities.md](docs/capabilities.md).

## Core capabilities

- centralized OpenAI settings and client creation;
- optional shared operation metadata and lifecycle observers;
- Responses API orchestration, structured outputs, files, tools, and websocket helpers;
- Agents SDK wrappers, runners, search workflows, and reusable text agents;
- typed Pydantic structures and Jinja prompt rendering;
- file and vector-store helpers;
- output validation and shared tool contracts;
- deterministic Codex plugin registration, discovery, inspection, and lifecycle;
- optional LangExtract and Streamlit integrations;
- local CLI inspection without hidden API calls.

Detailed maturity, installation, execution, and escape-hatch information lives
only in the [capability matrix](docs/capabilities.md).

## Configuration

`OpenAISettings` loads standard configuration from environment variables or a
local `.env` file and creates official SDK clients:

```python
from openai_sdk_helpers import OpenAISettings

settings = OpenAISettings.from_env()
client = settings.create_client()
```

Common variables include `OPENAI_API_KEY`, `OPENAI_ORG_ID`,
`OPENAI_PROJECT_ID`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_TIMEOUT`, and
`OPENAI_MAX_RETRIES`. Uncommon official client parameters remain available
through `extra_client_kwargs`.

## Security

Report suspected vulnerabilities through the repository's confidential GitHub
private vulnerability reporting flow, not a public issue. Never include real
credentials, customer prompts, model responses, uploaded files, or production
logs. See [SECURITY.md](SECURITY.md) for supported versions, reporting guidance,
and trust boundaries.

## Documentation

- [Capability matrix](docs/capabilities.md) — canonical feature inventory and maturity
- [Public API](docs/public-api.md) — intentional import surface
- [Operation context](docs/operation-context.md) — lifecycle hooks, usage, diagnostics, and SDK boundaries
- [Installation profiles](docs/installation.md) — core and optional dependencies
- [Supported examples](examples/README.md) — executable and illustrative example policy
- [Security policy](SECURITY.md) — confidential reporting and supported versions
- [Release checklist](docs/release-checklist.md) — security and publication gates
- [Codex plugins](docs/codex-plugins.md) — protocol, lifecycle, discovery, and packaging
- [0.8 Codex migration](docs/codex-plugin-migration-0.8.md) — compatibility guidance
- [Publishing](docs/publishing.md) — OIDC release process and recovery
- [Product vision](PRODUCT.md) — mission, users, principles, and non-goals
- [Roadmap](ROADMAP.md) — shipped foundations and release gates
- [Changelog](CHANGELOG.md) — user-visible changes

## Development

```bash
git clone https://github.com/fatmambot33/openai-sdk-helpers.git
cd openai-sdk-helpers
pip install -e ".[dev]"

pydocstyle src
black --check --diff .
pyright src
pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
python scripts/check_markdown_links.py
```

Additional CI validates Python 3.10–3.13, minimum and latest compatible OpenAI
SDK versions, built distributions, installed entry points, supported examples,
and isolated `core`, `extract`, `ui`, and `all` profiles.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before changing
public behavior.

## Scope boundaries

This project is not an end-user application, hosted platform, replacement SDK,
universal agent framework, prompt catalog, or storage service. Application-
specific business logic belongs in consuming projects.

MCP, consolidated retrieval, and Realtime helpers are roadmap items, not current
package promises. Their issue order and release gates are tracked in
[ROADMAP.md](ROADMAP.md).

## License

Licensed under the [MIT License](LICENSE).

---
name: openai-sdk-helpers
description: Install and use openai-sdk-helpers from Git to build typed OpenAI Responses API, agent, prompt, validation, extraction, and vector storage workflows.
---

# OpenAI SDK Helpers

## Setup

Ensure the current Python environment has the repository version installed:

```bash
python -m pip install --upgrade "git+https://github.com/fatmambot33/openai-sdk-helpers.git"
```

Use the package's Codex registry and public typed helpers rather than recreating orchestration utilities.

## Workflow

1. Inspect the installed package version and public exports.
2. Use `CodexPluginRegistry` to discover commands and capabilities.
3. Prefer structured output models and validators for machine-consumed responses.
4. Keep credentials in environment variables and never print secrets.
5. Use the Responses API helpers for new OpenAI integrations.
6. Run relevant tests and type checks after generated changes.

## Example

```python
from openai_sdk_helpers import CodexPluginRegistry

registry = CodexPluginRegistry()
print(registry.list_commands())
```

Follow the repository's NumPy-style docstring, Pyright, Black, and pytest conventions when editing code.

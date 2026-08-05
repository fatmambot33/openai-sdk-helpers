---
name: openai-sdk-helpers
description: Install and use openai-sdk-helpers from Git with OpenAI credentials loaded only from a local .env file for typed Responses API, agent, prompt, validation, extraction, and vector workflows.
---

# OpenAI SDK Helpers

## Local-only credential policy

Use only credentials stored in the user's local project `.env`. Never request, upload, persist, or copy API keys into Codex, Git, plugin files, chat messages, hosted configuration, MCP URLs, or command history. Never print secret values.

Before the first OpenAI API operation:

1. Check whether `.env` exists in the current working directory.
2. If it is missing, copy the included template:

```bash
cp plugins/openai-sdk-helpers/.env.example .env
```

3. Ask the user to edit `.env` locally:

```dotenv
OPENAI_API_KEY=
OPENAI_PROJECT=
OPENAI_ORG_ID=
```

Only `OPENAI_API_KEY` is required. Project and organization values are optional.

4. Ensure `.env` is ignored by Git. Add it to `.gitignore` when necessary.
5. Verify only that required variable names are present and non-empty. Never display values.
6. Stop before any network call until local configuration is complete.

Load `.env` locally with `python-dotenv` or the package configuration path. Do not fall back to Codex-managed credentials, hosted secrets, remote secret stores, or keys supplied in prompts.

## Setup

Install the repository version locally:

```bash
python -m pip install --upgrade "git+https://github.com/fatmambot33/openai-sdk-helpers.git"
```

Use the package's Codex registry and typed helpers rather than recreating orchestration utilities.

## Workflow

1. Complete the local `.env` configuration check.
2. Inspect the installed package version and public exports.
3. Use `CodexPluginRegistry` to discover commands and capabilities.
4. Prefer structured outputs and validators for machine-consumed responses.
5. Use Responses API helpers for new OpenAI integrations.
6. Redact keys, authorization headers, tokens, and secret-bearing errors.
7. Run relevant tests and type checks after generated changes.

## Example

```python
from dotenv import load_dotenv
from openai_sdk_helpers import CodexPluginRegistry

load_dotenv(".env")
registry = CodexPluginRegistry()
print(registry.list_commands())
```

Follow the repository's NumPy-style docstring, Pyright, Black, and pytest conventions when editing code.

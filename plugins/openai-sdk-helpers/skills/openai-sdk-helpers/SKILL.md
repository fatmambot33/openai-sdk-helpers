---
name: openai-sdk-helpers
description: Install and use openai-sdk-helpers from Git with OpenAI credentials loaded only from a local .env file.
---

# OpenAI SDK Helpers

## Local-only credential policy

Use only credentials stored in the user's local project `.env`. Never request secret values in chat, upload them, print them, or copy them into Codex, Git, plugin files, hosted configuration, MCP URLs, or command history.

## First-use setup

Install the repository version locally:

```bash
python -m pip install --upgrade "git+https://github.com/fatmambot33/openai-sdk-helpers.git"
```

Before any OpenAI network operation, run:

```bash
openai-helpers-credentials doctor
```

When the check fails, guide the user through:

```bash
openai-helpers-credentials configure
```

The wizard securely prompts for `OPENAI_API_KEY` without echoing it and optionally accepts `OPENAI_PROJECT` and `OPENAI_ORG_ID`. It writes only to the current project's `.env`, applies restrictive permissions where supported, refuses accidental overwrites, and warns when Git-ignore protection is missing.

Run the doctor again after setup. It validates file existence, required variable names, permissions, and `.gitignore` coverage without displaying credential values. Stop network operations until it passes.

Use `--env-file PATH` only for a deliberate non-default local file. Never fall back to hosted secrets, Codex-managed credentials, remote secret stores, or credentials supplied in prompts.

## Workflow

1. Run `openai-helpers-credentials doctor`.
2. Load `.env` locally with `python-dotenv`.
3. Use the package's Codex registry and typed helpers.
4. Prefer structured outputs and validators.
5. Redact credentials and authorization headers from outputs and errors.
6. Run relevant tests and type checks after code changes.

```python
from dotenv import load_dotenv
from openai_sdk_helpers import CodexPluginRegistry

load_dotenv(".env")
registry = CodexPluginRegistry()
print(registry.list_commands())
```

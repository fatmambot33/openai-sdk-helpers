# Migration guide for 0.8

Version 0.8.0 ships the completed Codex plugin foundation and production
hardening, while making extraction and UI integrations optional. Most core
Responses and Agents users require no code changes.

## Installation profiles

LangExtract and Streamlit are no longer installed by the base package.

Core-only applications continue to use:

```bash
pip install openai-sdk-helpers
```

Applications using document extraction must install:

```bash
pip install "openai-sdk-helpers[extract]"
```

Applications using `openai_sdk_helpers.streamlit_app` must install:

```bash
pip install "openai-sdk-helpers[ui]"
```

Applications needing both may use:

```bash
pip install "openai-sdk-helpers[all]"
```

Existing extraction imports remain supported when the `extract` profile is
installed. Accessing an extraction export without LangExtract now raises an
actionable `ImportError` naming the required installation command.

Development environments should continue to use `pip install -e ".[dev]"`,
which includes both optional integrations and all repository checks.

## Codex plugins

Existing plugins that implement only `name` and `setup(context)` continue to
work. No metadata migration is mandatory.

Recommended changes for plugin packages:

- declare `openai_sdk_helpers.codex` entry points;
- add `CodexPluginMetadata` for inspectable identity, version, capabilities, and
  deprecation state;
- use isolated discovery for third-party plugin environments;
- use startup and shutdown hooks for explicitly owned resources;
- declare asynchronous commands directly rather than wrapping them in a second
  synchronous registry;
- inspect installed plugins and commands with the package CLI before deployment.

See [codex-plugin-migration-0.8.md](codex-plugin-migration-0.8.md) for examples and
detailed plugin guidance.

## Response tool validation

`ResponseConfiguration` validates `tools` when the configuration is created.
The value must be a non-string sequence of mapping objects. String-like
containers and non-mapping entries now fail immediately with `TypeError` rather
than failing later during request construction.

Applications passing valid SDK-shaped tool dictionaries require no change.
Applications that supplied serialized JSON strings should deserialize them into
mapping objects before constructing the configuration.

## Publishing and supply chain

The repository release workflow now uses PyPI Trusted Publishing exclusively.
Maintainers must configure the documented PyPI publisher and protected GitHub
`pypi` environment. The workflow intentionally has no `PYPI_API_TOKEN` fallback.

A manual workflow dispatch with publication disabled performs the supported
build-only rehearsal. See [publishing.md](publishing.md) and
[release-checklist.md](release-checklist.md).

## Verification

After upgrading, verify the installation profile used by the application:

```bash
python -m pip check
openai-helpers --help
openai-helpers codex plugins
openai-helpers codex commands
```

Extraction applications should import their existing extraction entry points in
a clean environment containing the `extract` profile. Streamlit applications
should do the same with the `ui` profile.

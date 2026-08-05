# Codex plugin migration guide for 0.8

The 0.8 hardening work is backward compatible for existing plugins. A plugin that only defines `name` and `setup(context)` requires no code change.

## Recommended metadata

Add structured metadata so hosts can inspect identity and capabilities without executing commands:

```python
from openai_sdk_helpers.codex import CodexPluginMetadata


class MyPlugin:
    name = "my-plugin"
    metadata = CodexPluginMetadata(
        name=name,
        version="1.4.0",
        summary="My reusable Codex workflows.",
        capabilities=("responses", "file-search"),
    )
```

The metadata name must match `plugin.name`. Keep `api_version` at its default unless the plugin intentionally targets a newer contract.

## Safer host discovery

Existing hosts may keep fail-fast discovery:

```python
registry.discover()
```

Hosts that load optional or third-party packages should migrate to isolated discovery:

```python
report = registry.discover_isolated()
if not report.ok:
    for failure in report.failures:
        logger.warning(
            "Codex plugin %s failed: %s: %s",
            failure.entry_point,
            failure.error_type,
            failure.message,
        )
```

Healthy plugins remain registered even when another entry point fails.

## Operational inspection

Use `registry.inspect_plugins()` for deterministic plugin metadata and command ownership. Operators can also use:

```console
openai-helpers codex plugins
openai-helpers codex commands
```

## Deprecation

Set `deprecated=True` in plugin metadata before removal. Document the replacement and keep the deprecated plugin functional for at least one minor release whenever practical.

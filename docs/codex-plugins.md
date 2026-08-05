# Codex plugins

`openai-sdk-helpers` provides a deliberately small plugin contract for reusable Codex workflows. The registry handles discovery, command routing, lifecycle hooks, sync or async execution, and inspection without hiding OpenAI SDK calls.

## Create a plugin

```python
from openai_sdk_helpers.codex import (
    CodexPluginContext,
    CodexPluginMetadata,
)


class MyPlugin:
    name = "my-plugin"
    metadata = CodexPluginMetadata(
        name=name,
        version="1.0.0",
        summary="Example Codex workflow commands.",
        capabilities=("responses", "tools"),
    )

    def setup(self, context: CodexPluginContext) -> None:
        context.add_command("hello", lambda name: f"Hello, {name}!")

    async def startup(self) -> None:
        # Allocate optional resources.
        pass

    async def shutdown(self) -> None:
        # Release optional resources.
        pass
```

Only `name` and `setup(context)` are required. Structured `metadata` is optional, so plugins written against the original contract remain valid. `startup()` and `shutdown()` are optional and may be synchronous or asynchronous.

## Run plugins directly

```python
from openai_sdk_helpers import CodexPluginRegistry

registry = CodexPluginRegistry(metadata={"environment": "production"})
registry.register(MyPlugin())

async with registry:
    result = await registry.run_async("hello", "Codex")
```

Use `run()` for synchronous commands. It intentionally returns an awaitable unchanged when the registered command is asynchronous. Use `run_async()` when the command type is not known by the caller.

## Inspect capabilities

```python
for plugin in registry.inspect_plugins():
    print(plugin.metadata.name)
    print(plugin.metadata.version)
    print(plugin.metadata.capabilities)
    print(plugin.command_names)
```

Capability identifiers are plugin-defined, stable strings. They describe what a plugin exposes; they do not grant permissions or trigger network calls.

## Publish an installable plugin

Declare the plugin in the package's `pyproject.toml`:

```toml
[project.entry-points."openai_sdk_helpers.codex"]
my-plugin = "my_package.plugin:MyPlugin"
```

Then discover installed plugins:

```python
registry = CodexPluginRegistry()
registry.discover()
```

An entry point may expose either a plugin instance or a plugin class with a no-argument constructor.

## Isolate discovery failures

`discover()` keeps its original fail-fast behavior. Production hosts that load third-party plugins should use the isolated API:

```python
report = registry.discover_isolated()

for failure in report.failures:
    print(failure.entry_point, failure.error_type, failure.message)
```

A broken package cannot prevent later entry points from loading. The report contains successful plugins and deterministic failure records.

## CLI inspection

```console
openai-helpers codex plugins
openai-helpers codex commands
```

Both commands use isolated discovery. They return a non-zero exit code when any installed plugin fails to load while still printing healthy plugins and commands.

## Compatibility and deprecation policy

The current plugin contract version is `1`, exposed as `CODEX_PLUGIN_API_VERSION`.

- The required `name` and `setup(context)` protocol remains backward compatible throughout the `0.x` package line unless a security or correctness issue makes that impossible.
- Optional metadata fields and new inspection methods may be added without changing the plugin API version.
- A plugin that declares an unsupported `api_version` is rejected before setup, with a clear compatibility error.
- A planned removal must be documented for at least one minor release before removal.
- Deprecated plugins should set `CodexPluginMetadata(deprecated=True)` and provide migration guidance in their own documentation.
- `discover()` remains fail-fast for compatibility. Safer behavior is additive through `discover_isolated()`.
- Command names and capability identifiers are owned by plugins. Renaming or removing them is a plugin-level breaking change.

See [the 0.8 migration guide](codex-plugin-migration-0.8.md) for adoption guidance.

## Registration guarantees

- Plugin and command names must be unique and non-empty.
- Plugin setup is atomic: commands added by a failing setup are rolled back.
- Plugins cannot be added after registry startup.
- Startup runs in registration order.
- Shutdown runs in reverse registration order.
- Repeated startup and shutdown calls are safe no-ops.

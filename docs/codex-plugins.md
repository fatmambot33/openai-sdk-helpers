# Codex plugins

`openai-sdk-helpers` provides a deliberately small plugin contract for reusable Codex workflows. The registry handles discovery, command routing, lifecycle hooks, and sync or async execution without hiding OpenAI SDK calls.

## Create a plugin

```python
from openai_sdk_helpers import CodexPluginContext


class MyPlugin:
    name = "my-plugin"

    def setup(self, context: CodexPluginContext) -> None:
        context.add_command("hello", lambda name: f"Hello, {name}!")

    async def startup(self) -> None:
        # Allocate optional resources.
        pass

    async def shutdown(self) -> None:
        # Release optional resources.
        pass
```

Only `name` and `setup(context)` are required. `startup()` and `shutdown()` are optional and may be synchronous or asynchronous.

## Run plugins directly

```python
from openai_sdk_helpers import CodexPluginRegistry

registry = CodexPluginRegistry(metadata={"environment": "production"})
registry.register(MyPlugin())

async with registry:
    result = await registry.run_async("hello", "Codex")
```

Use `run()` for synchronous commands. It intentionally returns an awaitable unchanged when the registered command is asynchronous. Use `run_async()` when command type is not known by the caller.

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

## Registration guarantees

- Plugin and command names must be unique and non-empty.
- Plugin setup is atomic: commands added by a failing setup are rolled back.
- Plugins cannot be added after registry startup.
- Startup runs in registration order.
- Shutdown runs in reverse registration order.
- Repeated startup and shutdown calls are safe no-ops.

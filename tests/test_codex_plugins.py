"""Tests for the Codex plugin surface."""

from __future__ import annotations

import pytest

from openai_sdk_helpers.codex import CodexPluginContext, CodexPluginRegistry


class GreetingPlugin:
    """Small test plugin."""

    name = "greeting"

    def setup(self, context: CodexPluginContext) -> None:
        """Register a greeting command."""

        prefix = str(context.metadata.get("prefix", "hello"))
        context.add_command("greet", lambda name: f"{prefix} {name}")


def test_register_and_run_plugin_command() -> None:
    registry = CodexPluginRegistry(metadata={"prefix": "hi"})

    plugin = registry.register(GreetingPlugin())

    assert plugin.name == "greeting"
    assert registry.plugin_names == ("greeting",)
    assert registry.command_names == ("greet",)
    assert registry.run("greet", "Codex") == "hi Codex"


@pytest.mark.asyncio
async def test_run_async_supports_sync_and_async_commands() -> None:
    registry = CodexPluginRegistry()

    class MixedPlugin:
        name = "mixed"

        def setup(self, context: CodexPluginContext) -> None:
            context.add_command("sync", lambda value: value + 1)

            async def async_command(value: int) -> int:
                return value + 2

            context.add_command("async", async_command)

    registry.register(MixedPlugin())

    assert await registry.run_async("sync", 1) == 2
    assert await registry.run_async("async", 1) == 3


@pytest.mark.asyncio
async def test_lifecycle_hooks_and_context_manager() -> None:
    events: list[str] = []
    registry = CodexPluginRegistry()

    class LifecyclePlugin:
        name = "lifecycle"

        def setup(self, context: CodexPluginContext) -> None:
            context.add_command("status", lambda: "ready")

        async def startup(self) -> None:
            events.append("start")

        def shutdown(self) -> None:
            events.append("stop")

    registry.register(LifecyclePlugin())

    async with registry:
        assert registry.started is True
        assert registry.run("status") == "ready"

    assert registry.started is False
    assert events == ["start", "stop"]


def test_failed_setup_rolls_back_registered_commands() -> None:
    registry = CodexPluginRegistry()

    class BrokenPlugin:
        name = "broken"

        def setup(self, context: CodexPluginContext) -> None:
            context.add_command("temporary", lambda: None)
            raise RuntimeError("setup failed")

    with pytest.raises(RuntimeError, match="setup failed"):
        registry.register(BrokenPlugin())

    assert registry.plugin_names == ()
    assert registry.command_names == ()


def test_registration_after_startup_is_rejected() -> None:
    registry = CodexPluginRegistry()
    registry._started = True

    with pytest.raises(RuntimeError, match="after startup"):
        registry.register(GreetingPlugin())


def test_duplicate_plugin_is_rejected() -> None:
    registry = CodexPluginRegistry()
    registry.register(GreetingPlugin())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(GreetingPlugin())


def test_duplicate_command_is_rejected() -> None:
    registry = CodexPluginRegistry()
    registry.register(GreetingPlugin())

    class DuplicateCommandPlugin:
        name = "duplicate"

        def setup(self, context: CodexPluginContext) -> None:
            context.add_command("greet", lambda: None)

    with pytest.raises(ValueError, match="Command already registered"):
        registry.register(DuplicateCommandPlugin())


def test_unknown_command_has_clear_error() -> None:
    registry = CodexPluginRegistry()

    with pytest.raises(KeyError, match="Unknown Codex command"):
        registry.run("missing")


def test_invalid_plugin_is_rejected() -> None:
    registry = CodexPluginRegistry()

    with pytest.raises(TypeError, match="Plugin must define"):
        registry.register(object())  # type: ignore[arg-type]

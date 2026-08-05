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

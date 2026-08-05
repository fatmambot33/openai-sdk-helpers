"""Production-hardening tests for the Codex plugin surface."""

from __future__ import annotations

import importlib
from importlib.metadata import EntryPoint

import pytest

from openai_sdk_helpers.cli import main
from openai_sdk_helpers.codex import (
    CodexPluginContext,
    CodexPluginMetadata,
    CodexPluginRegistry,
)


class MetadataPlugin:
    """Plugin with structured metadata for inspection tests."""

    name = "metadata"
    metadata = CodexPluginMetadata(
        name="metadata",
        version="1.2.3",
        summary="Metadata test plugin.",
        capabilities=("responses", "tools", "tools"),
    )

    def setup(self, context: CodexPluginContext) -> None:
        """Register one command."""
        context.add_command("metadata.hello", lambda: "hello")


def test_metadata_and_capability_inspection() -> None:
    registry = CodexPluginRegistry()
    registry.register(MetadataPlugin())

    inspection = registry.inspect_plugins()[0]

    assert inspection.metadata.version == "1.2.3"
    assert inspection.metadata.capabilities == ("responses", "tools")
    assert inspection.command_names == ("metadata.hello",)


def test_plugin_without_metadata_remains_supported() -> None:
    class LegacyPlugin:
        name = "legacy"

        def setup(self, context: CodexPluginContext) -> None:
            context.add_command("legacy.run", lambda: None)

    registry = CodexPluginRegistry()
    registry.register(LegacyPlugin())

    metadata = registry.get_plugin_metadata("legacy")
    assert metadata.name == "legacy"
    assert metadata.version == "0"


def test_incompatible_plugin_api_version_is_rejected() -> None:
    class IncompatiblePlugin:
        name = "incompatible"
        metadata = CodexPluginMetadata(name=name, api_version="2")

        def setup(self, context: CodexPluginContext) -> None:
            context.add_command("incompatible.run", lambda: None)

    with pytest.raises(ValueError, match="Unsupported Codex plugin API version"):
        CodexPluginRegistry().register(IncompatiblePlugin())


def test_isolated_discovery_reports_failure_and_keeps_success() -> None:
    registry = CodexPluginRegistry()
    good = EntryPoint(
        name="metadata",
        value=f"{__name__}:MetadataPlugin",
        group="test.codex",
    )
    bad = EntryPoint(
        name="broken",
        value="missing_codex_plugin:Plugin",
        group="test.codex",
    )

    report = registry.load_entry_points_isolated((bad, good))

    assert registry.plugin_names == ("metadata",)
    assert len(report.plugins) == 1
    assert report.failures[0].entry_point == "broken"
    assert report.failures[0].error_type == "ModuleNotFoundError"
    assert report.ok is False


def test_installed_entry_point_discovery_end_to_end(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "installed_codex_plugin.py"
    module.write_text(
        "from openai_sdk_helpers.codex import CodexPluginContext\n"
        "class InstalledPlugin:\n"
        "    name = 'installed'\n"
        "    def setup(self, context: CodexPluginContext) -> None:\n"
        "        context.add_command('installed.run', lambda: 'installed')\n",
        encoding="utf-8",
    )
    dist_info = tmp_path / "installed_codex_plugin-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: installed-codex-plugin\nVersion: 1.0\n",
        encoding="utf-8",
    )
    group = "openai_sdk_helpers.codex.test"
    (dist_info / "entry_points.txt").write_text(
        f"[{group}]\ninstalled = installed_codex_plugin:InstalledPlugin\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    registry = CodexPluginRegistry()
    plugins = registry.discover(group=group)

    assert plugins[0].name == "installed"
    assert registry.run("installed.run") == "installed"


def test_codex_cli_lists_plugins_and_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def discover(self: CodexPluginRegistry, group: str = ""):
        del group
        self.register(MetadataPlugin())
        return self.load_entry_points_isolated(())

    monkeypatch.setattr(CodexPluginRegistry, "discover_isolated", discover)

    assert main(["codex", "plugins"]) == 0
    plugin_output = capsys.readouterr().out
    assert "metadata 1.2.3" in plugin_output
    assert "responses, tools" in plugin_output

    assert main(["codex", "commands"]) == 0
    command_output = capsys.readouterr().out
    assert "metadata.hello (metadata)" in command_output

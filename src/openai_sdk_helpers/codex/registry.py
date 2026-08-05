"""Registration and discovery for Codex plugins."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Iterable, Mapping

from .plugin import (
    CODEX_PLUGIN_API_VERSION,
    CodexCommand,
    CodexPlugin,
    CodexPluginContext,
    CodexPluginMetadata,
)

CODEX_PLUGIN_ENTRY_POINT = "openai_sdk_helpers.codex"


@dataclass(frozen=True, slots=True)
class CodexPluginInspection:
    """Inspectable plugin metadata and registered commands."""

    metadata: CodexPluginMetadata
    command_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CodexPluginDiscoveryFailure:
    """Serializable failure captured while loading one entry point."""

    entry_point: str
    value: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class CodexPluginDiscoveryReport:
    """Successful plugins and isolated discovery failures."""

    plugins: tuple[CodexPlugin, ...]
    failures: tuple[CodexPluginDiscoveryFailure, ...]

    @property
    def ok(self) -> bool:
        """Return whether every entry point loaded successfully."""
        return not self.failures


class CodexPluginRegistry:
    """Small, deterministic registry for Codex plugins and commands.

    Parameters
    ----------
    metadata
        Shared metadata exposed to every plugin during setup.
    """

    def __init__(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._plugins: dict[str, CodexPlugin] = {}
        self._plugin_metadata: dict[str, CodexPluginMetadata] = {}
        self._plugin_commands: dict[str, tuple[str, ...]] = {}
        self._commands: dict[str, CodexCommand] = {}
        self._context = CodexPluginContext(
            commands=self._commands,
            metadata={} if metadata is None else dict(metadata),
        )
        self._started = False

    @property
    def plugin_names(self) -> tuple[str, ...]:
        """Return registered plugin names in registration order."""
        return tuple(self._plugins)

    @property
    def command_names(self) -> tuple[str, ...]:
        """Return registered command names in registration order."""
        return tuple(self._commands)

    @property
    def started(self) -> bool:
        """Return whether plugin startup hooks have completed."""
        return self._started

    def register(self, plugin: CodexPlugin) -> CodexPlugin:
        """Register and initialize one plugin atomically.

        Parameters
        ----------
        plugin
            Object implementing :class:`CodexPlugin`.

        Returns
        -------
        CodexPlugin
            The registered plugin.

        Raises
        ------
        TypeError
            If the object does not implement the plugin protocol.
        ValueError
            If its name is empty, duplicated, or incompatible.
        RuntimeError
            If registration is attempted after startup.
        """
        if self._started:
            raise RuntimeError("Plugins cannot be registered after startup.")
        if not isinstance(plugin, CodexPlugin):
            raise TypeError("Plugin must define a name and setup(context).")
        name = plugin.name.strip()
        if not name:
            raise ValueError("Plugin name must not be empty.")
        if name in self._plugins:
            raise ValueError(f"Plugin already registered: {name}")

        metadata = self._resolve_metadata(plugin, name)
        previous_commands = dict(self._commands)
        try:
            plugin.setup(self._context)
        except Exception:
            self._commands.clear()
            self._commands.update(previous_commands)
            raise

        command_names = tuple(
            command_name
            for command_name in self._commands
            if command_name not in previous_commands
        )
        self._plugins[name] = plugin
        self._plugin_metadata[name] = metadata
        self._plugin_commands[name] = command_names
        return plugin

    def get_plugin(self, name: str) -> CodexPlugin:
        """Return a registered plugin by name."""
        return self._plugins[name]

    def get_plugin_metadata(self, name: str) -> CodexPluginMetadata:
        """Return normalized metadata for a registered plugin."""
        return self._plugin_metadata[name]

    def inspect_plugins(self) -> tuple[CodexPluginInspection, ...]:
        """Return deterministic metadata and command capability inspection."""
        return tuple(
            CodexPluginInspection(
                metadata=self._plugin_metadata[name],
                command_names=self._plugin_commands[name],
            )
            for name in self._plugins
        )

    def run(self, command: str, /, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered command.

        Async commands return an awaitable. Use :meth:`run_async` when the
        caller wants one API that supports both synchronous and asynchronous
        commands.
        """
        try:
            handler = self._commands[command]
        except KeyError as exc:
            raise KeyError(f"Unknown Codex command: {command}") from exc
        return handler(*args, **kwargs)

    async def run_async(self, command: str, /, *args: Any, **kwargs: Any) -> Any:
        """Execute a command and await its result when necessary."""
        result = self.run(command, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def startup(self) -> None:
        """Run optional plugin startup hooks in registration order."""
        if self._started:
            return
        for plugin in self._plugins.values():
            await self._call_hook(plugin, "startup")
        self._started = True

    async def shutdown(self) -> None:
        """Run optional plugin shutdown hooks in reverse registration order."""
        if not self._started:
            return
        for plugin in reversed(tuple(self._plugins.values())):
            await self._call_hook(plugin, "shutdown")
        self._started = False

    async def __aenter__(self) -> CodexPluginRegistry:
        """Start plugins and return this registry."""
        await self.startup()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Shut plugins down when leaving an async context."""
        await self.shutdown()

    def discover(
        self, group: str = CODEX_PLUGIN_ENTRY_POINT
    ) -> tuple[CodexPlugin, ...]:
        """Load and register plugins, preserving fail-fast compatibility."""
        discovered = entry_points().select(group=group)
        return self.load_entry_points(discovered)

    def discover_isolated(
        self, group: str = CODEX_PLUGIN_ENTRY_POINT
    ) -> CodexPluginDiscoveryReport:
        """Load installed plugins while isolating each entry-point failure."""
        discovered = entry_points().select(group=group)
        return self.load_entry_points_isolated(discovered)

    def load_entry_points(
        self, entry_point_values: Iterable[EntryPoint]
    ) -> tuple[CodexPlugin, ...]:
        """Load plugins from explicit entry points using fail-fast behavior."""
        loaded: list[CodexPlugin] = []
        for entry_point in entry_point_values:
            loaded.append(self._load_entry_point(entry_point))
        return tuple(loaded)

    def load_entry_points_isolated(
        self, entry_point_values: Iterable[EntryPoint]
    ) -> CodexPluginDiscoveryReport:
        """Load explicit entry points and report failures independently."""
        loaded: list[CodexPlugin] = []
        failures: list[CodexPluginDiscoveryFailure] = []
        for entry_point in entry_point_values:
            try:
                loaded.append(self._load_entry_point(entry_point))
            except Exception as exc:
                failures.append(
                    CodexPluginDiscoveryFailure(
                        entry_point=entry_point.name,
                        value=entry_point.value,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        return CodexPluginDiscoveryReport(tuple(loaded), tuple(failures))

    def _load_entry_point(self, entry_point: EntryPoint) -> CodexPlugin:
        candidate = entry_point.load()
        plugin = candidate() if isinstance(candidate, type) else candidate
        return self.register(plugin)

    @staticmethod
    def _resolve_metadata(plugin: CodexPlugin, name: str) -> CodexPluginMetadata:
        raw_metadata = getattr(plugin, "metadata", None)
        if raw_metadata is None:
            return CodexPluginMetadata(name=name)
        if not isinstance(raw_metadata, CodexPluginMetadata):
            raise TypeError("Plugin metadata must be CodexPluginMetadata.")
        if raw_metadata.name != name:
            raise ValueError("Plugin metadata name must match plugin name.")
        if raw_metadata.api_version != CODEX_PLUGIN_API_VERSION:
            raise ValueError(
                "Unsupported Codex plugin API version: "
                f"{raw_metadata.api_version}; expected {CODEX_PLUGIN_API_VERSION}."
            )
        return raw_metadata

    @staticmethod
    async def _call_hook(plugin: CodexPlugin, hook_name: str) -> None:
        hook = getattr(plugin, hook_name, None)
        if hook is None:
            return
        result = hook()
        if inspect.isawaitable(result):
            await result

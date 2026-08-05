"""Registration and discovery for Codex plugins."""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Any, Iterable, Mapping

from .plugin import CodexCommand, CodexPlugin, CodexPluginContext

CODEX_PLUGIN_ENTRY_POINT = "openai_sdk_helpers.codex"


class CodexPluginRegistry:
    """Small, deterministic registry for Codex plugins and commands.

    Parameters
    ----------
    metadata
        Shared metadata exposed to every plugin during setup.
    """

    def __init__(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._plugins: dict[str, CodexPlugin] = {}
        self._commands: dict[str, CodexCommand] = {}
        self._context = CodexPluginContext(
            commands=self._commands,
            metadata={} if metadata is None else dict(metadata),
        )

    @property
    def plugin_names(self) -> tuple[str, ...]:
        """Return registered plugin names in registration order."""

        return tuple(self._plugins)

    @property
    def command_names(self) -> tuple[str, ...]:
        """Return registered command names in registration order."""

        return tuple(self._commands)

    def register(self, plugin: CodexPlugin) -> CodexPlugin:
        """Register and initialize one plugin.

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
            If its name is empty or already registered.
        """

        if not isinstance(plugin, CodexPlugin):
            raise TypeError("Plugin must define a name and setup(context).")
        name = plugin.name.strip()
        if not name:
            raise ValueError("Plugin name must not be empty.")
        if name in self._plugins:
            raise ValueError(f"Plugin already registered: {name}")
        plugin.setup(self._context)
        self._plugins[name] = plugin
        return plugin

    def get_plugin(self, name: str) -> CodexPlugin:
        """Return a registered plugin by name."""

        return self._plugins[name]

    def run(self, command: str, /, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered command."""

        try:
            handler = self._commands[command]
        except KeyError as exc:
            raise KeyError(f"Unknown Codex command: {command}") from exc
        return handler(*args, **kwargs)

    def discover(self, group: str = CODEX_PLUGIN_ENTRY_POINT) -> tuple[CodexPlugin, ...]:
        """Load and register plugins exposed through package entry points."""

        discovered = entry_points().select(group=group)
        return self.load_entry_points(discovered)

    def load_entry_points(
        self, entry_point_values: Iterable[EntryPoint]
    ) -> tuple[CodexPlugin, ...]:
        """Load plugins from explicit entry points.

        This separate method keeps discovery easy to test without installed
        distributions.
        """

        loaded: list[CodexPlugin] = []
        for entry_point in entry_point_values:
            candidate = entry_point.load()
            plugin = candidate() if isinstance(candidate, type) else candidate
            loaded.append(self.register(plugin))
        return tuple(loaded)

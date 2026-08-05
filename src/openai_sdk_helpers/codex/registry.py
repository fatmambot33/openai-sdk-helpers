"""Registration and discovery for Codex plugins."""

from __future__ import annotations

import inspect
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
            If its name is empty or already registered.
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

        previous_commands = dict(self._commands)
        try:
            plugin.setup(self._context)
        except Exception:
            self._commands.clear()
            self._commands.update(previous_commands)
            raise

        self._plugins[name] = plugin
        return plugin

    def get_plugin(self, name: str) -> CodexPlugin:
        """Return a registered plugin by name."""
        return self._plugins[name]

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

    @staticmethod
    async def _call_hook(plugin: CodexPlugin, hook_name: str) -> None:
        hook = getattr(plugin, hook_name, None)
        if hook is None:
            return
        result = hook()
        if inspect.isawaitable(result):
            await result

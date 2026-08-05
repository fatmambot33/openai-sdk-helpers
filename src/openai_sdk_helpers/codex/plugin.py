"""Typed contracts for lightweight Codex plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

CODEX_PLUGIN_API_VERSION = "1"
CodexCommand = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class CodexPluginMetadata:
    """Structured identity and capabilities for a Codex plugin.

    Parameters
    ----------
    name
        Stable plugin name. It must match the plugin's ``name`` attribute.
    version
        Plugin package or implementation version.
    summary
        Short human-readable description.
    capabilities
        Stable capability identifiers exposed by the plugin.
    api_version
        Codex plugin contract version implemented by the plugin.
    deprecated
        Whether users should migrate away from the plugin.
    """

    name: str
    version: str = "0"
    summary: str = ""
    capabilities: tuple[str, ...] = ()
    api_version: str = CODEX_PLUGIN_API_VERSION
    deprecated: bool = False

    def __post_init__(self) -> None:
        """Normalize values and reject ambiguous metadata."""
        name = self.name.strip()
        version = self.version.strip()
        api_version = self.api_version.strip()
        if not name:
            raise ValueError("Plugin metadata name must not be empty.")
        if not version:
            raise ValueError("Plugin metadata version must not be empty.")
        if not api_version:
            raise ValueError("Plugin API version must not be empty.")

        capabilities: list[str] = []
        for capability in self.capabilities:
            normalized = capability.strip()
            if not normalized:
                raise ValueError("Plugin capabilities must not be empty.")
            if normalized not in capabilities:
                capabilities.append(normalized)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "summary", self.summary.strip())
        object.__setattr__(self, "api_version", api_version)
        object.__setattr__(self, "capabilities", tuple(capabilities))


@dataclass(frozen=True, slots=True)
class CodexPluginContext:
    """Context exposed to plugins during registration.

    Parameters
    ----------
    commands
        Mutable command mapping owned by the registry.
    metadata
        Shared read-only metadata for plugin setup.
    """

    commands: dict[str, CodexCommand]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def add_command(self, name: str, command: CodexCommand) -> None:
        """Register a command under a unique non-empty name.

        Parameters
        ----------
        name
            Public command name.
        command
            Callable invoked for the command.

        Raises
        ------
        ValueError
            If the name is empty or already registered.
        TypeError
            If ``command`` is not callable.
        """
        normalized = name.strip()
        if not normalized:
            raise ValueError("Command name must not be empty.")
        if normalized in self.commands:
            raise ValueError(f"Command already registered: {normalized}")
        if not callable(command):
            raise TypeError("Command must be callable.")
        self.commands[normalized] = command


@runtime_checkable
class CodexPlugin(Protocol):
    """Minimal protocol implemented by first-class Codex plugins."""

    name: str

    def setup(self, context: CodexPluginContext) -> None:
        """Register commands and capabilities on ``context``."""

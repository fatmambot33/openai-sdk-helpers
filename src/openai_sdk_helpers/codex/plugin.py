"""Typed contracts for lightweight Codex plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

CodexCommand = Callable[..., Any]


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

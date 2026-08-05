"""First-class, lightweight Codex plugin surface."""

from .plugin import (
    CODEX_PLUGIN_API_VERSION,
    CodexCommand,
    CodexPlugin,
    CodexPluginContext,
    CodexPluginMetadata,
)
from .registry import (
    CODEX_PLUGIN_ENTRY_POINT,
    CodexPluginDiscoveryFailure,
    CodexPluginDiscoveryReport,
    CodexPluginInspection,
    CodexPluginRegistry,
)

__all__ = [
    "CODEX_PLUGIN_API_VERSION",
    "CODEX_PLUGIN_ENTRY_POINT",
    "CodexCommand",
    "CodexPlugin",
    "CodexPluginContext",
    "CodexPluginDiscoveryFailure",
    "CodexPluginDiscoveryReport",
    "CodexPluginInspection",
    "CodexPluginMetadata",
    "CodexPluginRegistry",
]

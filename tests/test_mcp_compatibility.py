"""Compatibility tests for the focused MCP transport surface."""

from __future__ import annotations

import inspect

import pytest

from openai_sdk_helpers.mcp import HostedMCPConfig, StreamableHTTPMCPConfig


def test_supported_agents_sdk_exposes_required_mcp_constructor_fields() -> None:
    """Lock the minimum constructor fields used by the adapter layer."""
    from agents import HostedMCPTool
    from agents.mcp import MCPServerStreamableHttp

    hosted = inspect.signature(HostedMCPTool)
    streamable = inspect.signature(MCPServerStreamableHttp)

    assert "tool_config" in hosted.parameters
    assert "params" in streamable.parameters
    assert "name" in streamable.parameters
    assert "use_structured_content" in streamable.parameters


def test_hosted_config_rejects_non_string_approval_scalar() -> None:
    """Reject values outside the declared hosted approval contract."""
    with pytest.raises(TypeError, match="string or mapping"):
        HostedMCPConfig(
            "docs",
            "https://example.test/mcp",
            require_approval=True,  # type: ignore[arg-type]
        )


def test_streamable_config_rejects_non_string_header_values() -> None:
    """Reject malformed headers before official SDK object construction."""
    with pytest.raises(TypeError, match="header values must be strings"):
        StreamableHTTPMCPConfig(
            "https://example.test/mcp",
            headers={"X-Retry": 3},  # type: ignore[dict-item]
        )

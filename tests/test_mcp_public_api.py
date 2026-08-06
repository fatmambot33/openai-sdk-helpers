"""Contract tests for the focused MCP public surface."""

from __future__ import annotations

from openai_sdk_helpers import mcp

EXPECTED_MCP_API = (
    "HostedMCPConfig",
    "MCPServerProtocol",
    "MCPTransport",
    "ManagedMCPServer",
    "StreamableHTTPMCPConfig",
    "build_hosted_mcp_tool",
    "build_streamable_http_server",
)


def test_mcp_module_exports_are_explicit() -> None:
    assert tuple(mcp.__all__) == EXPECTED_MCP_API
    assert all(hasattr(mcp, name) for name in EXPECTED_MCP_API)

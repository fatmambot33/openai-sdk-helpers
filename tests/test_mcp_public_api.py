"""Contract tests for the focused MCP public surface."""

from __future__ import annotations

from openai_sdk_helpers import mcp

EXPECTED_MCP_API = (
    "HostedMCPConfig",
    "MCPApprovalDecision",
    "MCPApprovalHandler",
    "MCPApprovalRequest",
    "MCPRetryPolicy",
    "MCPServerFailure",
    "MCPServerProtocol",
    "MCPToolCache",
    "MCPToolCacheEntry",
    "MCPToolDescriptor",
    "MCPToolListReport",
    "MCPToolPolicy",
    "MCPTransport",
    "ManagedMCPServer",
    "StreamableHTTPMCPConfig",
    "build_agents_tool_filter",
    "build_hosted_mcp_tool",
    "build_streamable_http_server",
    "list_tools_isolated",
    "request_approval",
    "run_safe_tool_call",
)


def test_mcp_module_exports_are_explicit() -> None:
    assert tuple(mcp.__all__) == EXPECTED_MCP_API
    assert all(hasattr(mcp, name) for name in EXPECTED_MCP_API)

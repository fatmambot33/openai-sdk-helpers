"""Opt-in adapters for official OpenAI Agents SDK MCP transports."""

from .transport import (
    HostedMCPConfig,
    MCPServerProtocol,
    MCPTransport,
    ManagedMCPServer,
    StreamableHTTPMCPConfig,
    build_hosted_mcp_tool,
    build_streamable_http_server,
)

__all__ = [
    "HostedMCPConfig",
    "MCPServerProtocol",
    "MCPTransport",
    "ManagedMCPServer",
    "StreamableHTTPMCPConfig",
    "build_hosted_mcp_tool",
    "build_streamable_http_server",
]

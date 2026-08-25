"""Typed adapters for official hosted and Streamable HTTP MCP transports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, cast

from ..runtime import OperationContext, run_observed_async


class MCPTransport(str, Enum):
    """Official MCP transport families supported by this package."""

    HOSTED = "hosted"
    STREAMABLE_HTTP = "streamable_http"


@dataclass(frozen=True, slots=True)
class HostedMCPConfig:
    """Configuration for the official hosted MCP tool.

    Parameters
    ----------
    server_label : str
        Stable label exposed to the model and tool-call events.
    server_url : str
        Remote MCP server URL used by the OpenAI platform.
    require_approval : str or Mapping[str, Any], default="always"
        Official hosted MCP approval setting passed through unchanged.
    allowed_tools : tuple[str, ...], default=()
        Optional explicit tool allow-list.
    server_description : str or None, default=None
        Optional description sent with the hosted tool configuration.
    authorization : str or None, default=None
        Optional caller-provided authorization value. It is excluded from the
        dataclass representation and never copied into operation diagnostics.

    Methods
    -------
    as_tool_config()
        Return the official SDK-shaped hosted MCP tool configuration.
    """

    server_label: str
    server_url: str
    require_approval: str | Mapping[str, Any] = "always"
    allowed_tools: tuple[str, ...] = ()
    server_description: str | None = None
    authorization: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize identity and copied collection values."""
        label = _required(self.server_label, "server_label")
        url = _required(self.server_url, "server_url")
        tools = tuple(
            dict.fromkeys(
                _required(tool, "allowed_tool") for tool in self.allowed_tools
            )
        )
        object.__setattr__(self, "server_label", label)
        object.__setattr__(self, "server_url", url)
        object.__setattr__(self, "allowed_tools", tools)
        if isinstance(self.require_approval, Mapping):
            object.__setattr__(self, "require_approval", dict(self.require_approval))
        elif not isinstance(self.require_approval, str):
            raise TypeError("require_approval must be a string or mapping")
        elif not self.require_approval.strip():
            raise ValueError("require_approval must not be empty")
        if self.server_description is not None:
            description = self.server_description.strip()
            object.__setattr__(self, "server_description", description or None)
        if self.authorization is not None and not self.authorization.strip():
            raise ValueError("authorization must not be empty")

    def as_tool_config(self) -> dict[str, Any]:
        """Return the official SDK-shaped hosted MCP tool configuration.

        Returns
        -------
        dict[str, Any]
            Fresh mapping suitable for ``HostedMCPTool(tool_config=...)``.
        """
        config: dict[str, Any] = {
            "type": "mcp",
            "server_label": self.server_label,
            "server_url": self.server_url,
            "require_approval": (
                dict(self.require_approval)
                if isinstance(self.require_approval, Mapping)
                else self.require_approval
            ),
        }
        if self.allowed_tools:
            config["allowed_tools"] = list(self.allowed_tools)
        if self.server_description is not None:
            config["server_description"] = self.server_description
        if self.authorization is not None:
            config["authorization"] = self.authorization
        return config


@dataclass(frozen=True, slots=True)
class StreamableHTTPMCPConfig:
    """Configuration for the official Agents SDK Streamable HTTP server.

    Parameters
    ----------
    url : str
        Streamable HTTP MCP endpoint.
    name : str or None, default=None
        Optional local server display name.
    headers : Mapping[str, str], default={}
        Caller-owned request headers copied at construction. Values are excluded
        from the dataclass representation.
    timeout_seconds : float, default=5.0
        Request timeout passed to the MCP transport.
    sse_read_timeout_seconds : float, default=300.0
        Streaming read timeout passed to the transport.
    terminate_on_close : bool, default=True
        Whether the transport should request session termination on cleanup.
    use_structured_content : bool, default=False
        Preserve structured MCP content where supported by the Agents SDK.

    Methods
    -------
    as_params()
        Return official Streamable HTTP transport parameters.
    """

    url: str
    name: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    timeout_seconds: float = 5.0
    sse_read_timeout_seconds: float = 300.0
    terminate_on_close: bool = True
    use_structured_content: bool = False

    def __post_init__(self) -> None:
        """Normalize transport values and copy caller mappings."""
        object.__setattr__(self, "url", _required(self.url, "url"))
        if self.name is not None:
            name = self.name.strip()
            object.__setattr__(self, "name", name or None)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.sse_read_timeout_seconds <= 0:
            raise ValueError("sse_read_timeout_seconds must be positive")
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            normalized_key = _required(key, "header_name")
            if not isinstance(value, str):
                raise TypeError("header values must be strings")
            headers[normalized_key] = value
        object.__setattr__(self, "headers", headers)

    def as_params(self) -> dict[str, Any]:
        """Return official Streamable HTTP transport parameters.

        Returns
        -------
        dict[str, Any]
            Fresh parameters mapping for ``MCPServerStreamableHttp``.
        """
        return {
            "url": self.url,
            "headers": dict(self.headers),
            "timeout": self.timeout_seconds,
            "sse_read_timeout": self.sse_read_timeout_seconds,
            "terminate_on_close": self.terminate_on_close,
        }


class MCPServerProtocol(Protocol):
    """Minimal lifecycle exposed by official Agents SDK MCP servers.

    Methods
    -------
    connect()
        Connect the server transport.
    cleanup()
        Clean up the server transport.
    """

    async def connect(self) -> None:
        """Connect the server transport."""
        ...

    async def cleanup(self) -> None:
        """Clean up the server transport."""
        ...


class ManagedMCPServer:
    """Explicit lifecycle wrapper for one official Agents SDK MCP server.

    Parameters
    ----------
    server : MCPServerProtocol
        Underlying official server. The wrapper owns only its connection
        lifecycle, not credentials, application state, or remote server data.
    transport : MCPTransport
        Transport identity used for diagnostics.

    Methods
    -------
    connect(*, operation_context=None)
        Connect explicitly and return the original SDK server.
    cleanup(*, operation_context=None)
        Clean up explicitly; repeated cleanup is a no-op.
    """

    def __init__(self, server: MCPServerProtocol, transport: MCPTransport) -> None:
        self._server = server
        self._transport = transport
        self._connected = False

    @property
    def raw_server(self) -> MCPServerProtocol:
        """Return the underlying official Agents SDK server."""
        return self._server

    @property
    def transport(self) -> MCPTransport:
        """Return the configured transport family."""
        return self._transport

    @property
    def connected(self) -> bool:
        """Return whether this wrapper completed ``connect`` successfully."""
        return self._connected

    async def connect(
        self,
        *,
        operation_context: OperationContext | None = None,
    ) -> MCPServerProtocol:
        """Connect explicitly and return the original SDK server.

        Parameters
        ----------
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context.

        Returns
        -------
        MCPServerProtocol
            Underlying official server.

        Raises
        ------
        BaseException
            Original SDK connection error, cancellation, or process interrupt.
        """
        if self._connected:
            return self._server

        async def execute() -> MCPServerProtocol:
            await self._server.connect()
            self._connected = True
            return self._server

        return await run_observed_async(operation_context, execute)

    async def cleanup(
        self,
        *,
        operation_context: OperationContext | None = None,
    ) -> None:
        """Clean up explicitly; repeated cleanup is a no-op.

        Parameters
        ----------
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context.

        Raises
        ------
        BaseException
            Original SDK cleanup error, cancellation, or process interrupt.
        """
        if not self._connected:
            return

        async def execute() -> None:
            await self._server.cleanup()
            self._connected = False

        await run_observed_async(operation_context, execute)

    async def __aenter__(self) -> MCPServerProtocol:
        """Connect and return the original SDK server."""
        return await self.connect()

    async def __aexit__(self, *_: object) -> None:
        """Clean up the transport when leaving the async context."""
        await self.cleanup()


def build_hosted_mcp_tool(config: HostedMCPConfig) -> object:
    """Build the official Agents SDK hosted MCP tool without executing it.

    Parameters
    ----------
    config : HostedMCPConfig
        Explicit hosted MCP configuration.

    Returns
    -------
    object
        Original official Agents SDK ``HostedMCPTool`` instance.

    Raises
    ------
    ImportError
        If the installed MCP/Agents integration is unavailable.
    """
    hosted_tool_type = _load_hosted_mcp_tool_type()
    return hosted_tool_type(tool_config=config.as_tool_config())


def build_streamable_http_server(
    config: StreamableHTTPMCPConfig,
) -> ManagedMCPServer:
    """Build an unconnected official Streamable HTTP MCP server.

    Parameters
    ----------
    config : StreamableHTTPMCPConfig
        Explicit Streamable HTTP transport configuration.

    Returns
    -------
    ManagedMCPServer
        Lifecycle wrapper preserving access to the original SDK server.

    Raises
    ------
    ImportError
        If the installed MCP/Agents integration is unavailable.
    """
    server_type = _load_streamable_http_server_type()
    kwargs: dict[str, Any] = {
        "params": config.as_params(),
        "use_structured_content": config.use_structured_content,
    }
    if config.name is not None:
        kwargs["name"] = config.name
    server = cast(MCPServerProtocol, server_type(**kwargs))
    return ManagedMCPServer(server, MCPTransport.STREAMABLE_HTTP)


def _load_hosted_mcp_tool_type() -> type[Any]:
    try:
        from agents import HostedMCPTool
    except ImportError as error:
        raise ImportError(
            "Hosted MCP is unavailable. Install with "
            'pip install "openai-sdk-helpers[mcp]".'
        ) from error
    return HostedMCPTool


def _load_streamable_http_server_type() -> type[Any]:
    try:
        from agents.mcp import MCPServerStreamableHttp
    except ImportError as error:
        raise ImportError(
            "Streamable HTTP MCP is unavailable. Install with "
            'pip install "openai-sdk-helpers[mcp]".'
        ) from error
    return MCPServerStreamableHttp


def _required(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


__all__ = [
    "HostedMCPConfig",
    "MCPServerProtocol",
    "MCPTransport",
    "ManagedMCPServer",
    "StreamableHTTPMCPConfig",
    "build_hosted_mcp_tool",
    "build_streamable_http_server",
]

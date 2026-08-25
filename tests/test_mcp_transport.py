"""Tests for opt-in MCP transport adapters."""

from __future__ import annotations

from typing import Any

import pytest

from openai_sdk_helpers import OperationContext, OperationEvent, OperationPhase
from openai_sdk_helpers.mcp import (
    HostedMCPConfig,
    MCPTransport,
    StreamableHTTPMCPConfig,
    build_hosted_mcp_tool,
    build_streamable_http_server,
)
from openai_sdk_helpers.mcp import transport as transport_module


class FakeHostedTool:
    """SDK-shaped hosted MCP tool fixture."""

    def __init__(self, *, tool_config: dict[str, Any]) -> None:
        self.tool_config = tool_config


class FakeServer:
    """SDK-shaped Streamable HTTP MCP server fixture."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.connect_calls = 0
        self.cleanup_calls = 0
        self.connect_error: BaseException | None = None

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


class Collector:
    """Collect operation lifecycle events."""

    def __init__(self) -> None:
        self.events: list[OperationEvent] = []

    def __call__(self, event: OperationEvent) -> None:
        self.events.append(event)


def test_hosted_config_preserves_policy_and_hides_authorization_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport_module,
        "_load_hosted_mcp_tool_type",
        lambda: FakeHostedTool,
    )
    config = HostedMCPConfig(
        server_label="docs",
        server_url="https://example.test/mcp",
        require_approval="always",
        allowed_tools=("search", "search", "read"),
        server_description="Documentation server",
        authorization="Bearer placeholder",
    )

    tool = build_hosted_mcp_tool(config)

    assert isinstance(tool, FakeHostedTool)
    assert tool.tool_config == {
        "type": "mcp",
        "server_label": "docs",
        "server_url": "https://example.test/mcp",
        "require_approval": "always",
        "allowed_tools": ["search", "read"],
        "server_description": "Documentation server",
        "authorization": "Bearer placeholder",
    }
    assert "Bearer placeholder" not in repr(config)


def test_streamable_builder_forwards_explicit_transport_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport_module,
        "_load_streamable_http_server_type",
        lambda: FakeServer,
    )
    config = StreamableHTTPMCPConfig(
        url="https://example.test/mcp",
        name="docs",
        headers={"Authorization": "Bearer placeholder"},
        timeout_seconds=7,
        sse_read_timeout_seconds=120,
        terminate_on_close=False,
        use_structured_content=True,
    )

    managed = build_streamable_http_server(config)
    server = managed.raw_server

    assert isinstance(server, FakeServer)
    assert managed.transport is MCPTransport.STREAMABLE_HTTP
    assert managed.connected is False
    assert server.kwargs == {
        "params": {
            "url": "https://example.test/mcp",
            "headers": {"Authorization": "Bearer placeholder"},
            "timeout": 7,
            "sse_read_timeout": 120,
            "terminate_on_close": False,
        },
        "use_structured_content": True,
        "name": "docs",
    }
    assert "Bearer placeholder" not in repr(config)


@pytest.mark.asyncio
async def test_managed_server_lifecycle_is_explicit_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport_module,
        "_load_streamable_http_server_type",
        lambda: FakeServer,
    )
    managed = build_streamable_http_server(
        StreamableHTTPMCPConfig(url="https://example.test/mcp")
    )
    server = managed.raw_server

    assert isinstance(server, FakeServer)
    assert server.connect_calls == 0
    assert server.cleanup_calls == 0

    returned = await managed.connect()
    assert returned is server
    assert managed.connected is True
    await managed.connect()
    assert server.connect_calls == 1

    await managed.cleanup()
    assert managed.connected is False
    await managed.cleanup()
    assert server.cleanup_calls == 1


@pytest.mark.asyncio
async def test_async_context_returns_raw_server_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport_module,
        "_load_streamable_http_server_type",
        lambda: FakeServer,
    )
    managed = build_streamable_http_server(
        StreamableHTTPMCPConfig(url="https://example.test/mcp")
    )
    server = managed.raw_server

    async with managed as connected:
        assert connected is server
        assert managed.connected is True

    assert managed.connected is False
    assert isinstance(server, FakeServer)
    assert server.cleanup_calls == 1


@pytest.mark.asyncio
async def test_connect_failure_preserves_exception_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport_module,
        "_load_streamable_http_server_type",
        lambda: FakeServer,
    )
    collector = Collector()
    context = OperationContext("mcp.connect", observers=(collector,))
    managed = build_streamable_http_server(
        StreamableHTTPMCPConfig(url="https://example.test/mcp")
    )
    server = managed.raw_server
    assert isinstance(server, FakeServer)
    error = ConnectionError("server unavailable")
    server.connect_error = error

    with pytest.raises(ConnectionError) as exc_info:
        await managed.connect(operation_context=context)

    assert exc_info.value is error
    assert managed.connected is False
    assert [event.phase for event in collector.events] == [
        OperationPhase.START,
        OperationPhase.FAILURE,
    ]
    assert collector.events[-1].error is error


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HostedMCPConfig(" ", "https://example.test/mcp"),
        lambda: HostedMCPConfig("docs", " "),
        lambda: HostedMCPConfig("docs", "https://example.test/mcp", " "),
        lambda: StreamableHTTPMCPConfig(" "),
        lambda: StreamableHTTPMCPConfig("https://example.test/mcp", timeout_seconds=0),
        lambda: StreamableHTTPMCPConfig(
            "https://example.test/mcp",
            sse_read_timeout_seconds=0,
        ),
    ],
)
def test_mcp_configs_reject_invalid_values(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]

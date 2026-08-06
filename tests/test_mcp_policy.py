"""Tests for explicit MCP policy and resilience helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openai_sdk_helpers.mcp import (
    MCPApprovalDecision,
    MCPApprovalRequest,
    MCPRetryPolicy,
    MCPToolCache,
    MCPToolDescriptor,
    MCPToolPolicy,
    StreamableHTTPMCPConfig,
    build_agents_tool_filter,
    build_streamable_http_server,
    list_tools_isolated,
    request_approval,
    run_safe_tool_call,
)
from openai_sdk_helpers.mcp import transport as transport_module


def test_tool_policy_filters_in_server_order_and_blocks_override() -> None:
    policy = MCPToolPolicy(
        allowed_tools=("search", "read", "write"),
        blocked_tools=("write",),
        approval_tools=("read",),
        safe_retry_tools=("search",),
    )
    tools = (
        MCPToolDescriptor("read"),
        MCPToolDescriptor("search"),
        MCPToolDescriptor("write"),
        MCPToolDescriptor("unknown"),
    )

    visible = policy.filter(tools)

    assert [tool.name for tool in visible] == ["read", "search"]
    assert policy.requires_approval("read") is True
    assert policy.requires_approval("search") is False
    assert policy.permits_retry("search") is True
    assert policy.permits_retry("read") is False


def test_tool_policy_rejects_conflicting_configuration() -> None:
    with pytest.raises(ValueError, match="both allowed and blocked"):
        MCPToolPolicy(allowed_tools=("write",), blocked_tools=("write",))
    with pytest.raises(ValueError, match="blocked tools cannot require approval"):
        MCPToolPolicy(blocked_tools=("write",), approval_tools=("write",))


@pytest.mark.asyncio
async def test_approval_is_fail_closed_and_arguments_are_hidden() -> None:
    policy = MCPToolPolicy(
        allowed_tools=("read", "write"),
        approval_tools=("write",),
    )
    request = MCPApprovalRequest(
        server_label="docs",
        tool_name="write",
        arguments={"secret": "value"},
    )

    assert await request_approval(policy, request, None) is MCPApprovalDecision.REJECT
    assert "value" not in repr(request)

    async def approve(_: MCPApprovalRequest) -> MCPApprovalDecision:
        return MCPApprovalDecision.APPROVE

    assert (
        await request_approval(policy, request, approve)
        is MCPApprovalDecision.APPROVE
    )
    blocked = MCPApprovalRequest("docs", "delete", {})
    assert await request_approval(policy, blocked, approve) is MCPApprovalDecision.REJECT


@pytest.mark.asyncio
async def test_safe_retry_requires_explicit_idempotence_and_policy() -> None:
    policy = MCPToolPolicy(
        allowed_tools=("search", "write"),
        safe_retry_tools=("search",),
    )
    retry = MCPRetryPolicy(max_attempts=3, backoff_seconds=0.5)
    attempts = 0
    sleeps: list[float] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary")
        return "ok"

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = await run_safe_tool_call(
        "search",
        operation,
        tool_policy=policy,
        retry_policy=retry,
        idempotent=True,
        sleep=sleep,
    )

    assert result == "ok"
    assert attempts == 3
    assert sleeps == [0.5, 0.5]

    with pytest.raises(ValueError, match="Retries require"):
        await run_safe_tool_call(
            "write",
            lambda: "never",
            tool_policy=policy,
            retry_policy=retry,
            idempotent=False,
        )


def test_tool_cache_expires_and_invalidates_explicitly() -> None:
    now = 10.0

    def clock() -> float:
        return now

    cache = MCPToolCache(5, clock=clock)
    tools = (MCPToolDescriptor("search"),)

    cache.set("docs", tools)
    assert cache.get("docs") == tools

    now = 16.0
    assert cache.get("docs") is None

    cache.set("docs", tools)
    cache.invalidate("docs")
    assert cache.get("docs") is None

    cache.set("docs", tools)
    cache.set("other", tools)
    cache.invalidate()
    assert cache.get("docs") is None
    assert cache.get("other") is None


class ToolServer:
    """Server fixture returning a configured tool list."""

    def __init__(self, tools: tuple[object, ...]) -> None:
        self.tools = tools
        self.calls = 0

    async def list_tools(self) -> tuple[object, ...]:
        self.calls += 1
        return self.tools


class BrokenServer:
    """Server fixture raising one preserved error."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def list_tools(self) -> tuple[object, ...]:
        raise self.error


@pytest.mark.asyncio
async def test_tool_listing_isolates_failures_and_uses_cache() -> None:
    good = ToolServer(
        (
            SimpleNamespace(
                name="search",
                description="Search docs",
                inputSchema={"type": "object"},
            ),
            SimpleNamespace(name="write", description="Write docs", inputSchema={}),
        )
    )
    error = ConnectionError("offline")
    cache = MCPToolCache(60)
    policy = MCPToolPolicy(blocked_tools=("write",))

    first = await list_tools_isolated(
        {"good": good, "bad": BrokenServer(error)},
        policy=policy,
        cache=cache,
    )

    assert first.ok is False
    assert [tool.name for tool in first.tools["good"]] == ["search"]
    assert first.failures[0].server_label == "bad"
    assert first.failures[0].error is error
    assert first.cache_hits == ()

    second = await list_tools_isolated(
        {"good": good},
        policy=policy,
        cache=cache,
    )
    assert second.ok is True
    assert second.cache_hits == ("good",)
    assert good.calls == 1


def test_official_static_filter_is_built_from_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_filter(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    import agents.mcp

    monkeypatch.setattr(agents.mcp, "create_static_tool_filter", fake_filter)
    policy = MCPToolPolicy(
        allowed_tools=("search",),
        blocked_tools=("write",),
    )

    built = build_agents_tool_filter(policy)

    assert built is not None
    assert captured == {
        "allowed_tool_names": ["search"],
        "blocked_tool_names": ["write"],
    }


class FakeServer:
    """SDK-shaped server constructor fixture."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def connect(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None


def test_streamable_server_receives_explicit_cache_and_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport_module,
        "_load_streamable_http_server_type",
        lambda: FakeServer,
    )
    tool_filter = object()
    managed = build_streamable_http_server(
        StreamableHTTPMCPConfig(
            url="https://example.test/mcp",
            cache_tools_list=True,
            tool_filter=tool_filter,
        )
    )
    server = managed.raw_server

    assert isinstance(server, FakeServer)
    assert server.kwargs["cache_tools_list"] is True
    assert server.kwargs["tool_filter"] is tool_filter

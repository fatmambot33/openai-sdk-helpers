"""Explicit MCP filtering, approvals, caching, retries, and isolation."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, cast

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    """Normalized MCP tool identity with raw SDK access."""

    name: str
    description: str | None = None
    input_schema: Mapping[str, Any] | None = None
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize tool identity and copy the optional schema."""
        name = self.name.strip()
        if not name:
            raise ValueError("name must not be empty")
        object.__setattr__(self, "name", name)
        if self.description is not None:
            description = self.description.strip()
            object.__setattr__(self, "description", description or None)
        if self.input_schema is not None:
            object.__setattr__(self, "input_schema", dict(self.input_schema))


@dataclass(frozen=True, slots=True)
class MCPToolPolicy:
    """Deterministic tool visibility, approval, and retry policy.

    Parameters
    ----------
    allowed_tools : tuple[str, ...], default=()
        Optional allow-list. Empty means all non-blocked tools are visible.
    blocked_tools : tuple[str, ...], default=()
        Tools removed even when present in the allow-list.
    approval_tools : tuple[str, ...], default=()
        Visible tools that always require explicit application approval.
    safe_retry_tools : tuple[str, ...], default=()
        Tools the application has explicitly classified as safe for retry.
    """

    allowed_tools: tuple[str, ...] = ()
    blocked_tools: tuple[str, ...] = ()
    approval_tools: tuple[str, ...] = ()
    safe_retry_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize and validate deterministic policy sets."""
        allowed = _names(self.allowed_tools, "allowed_tool")
        blocked = _names(self.blocked_tools, "blocked_tool")
        approvals = _names(self.approval_tools, "approval_tool")
        retryable = _names(self.safe_retry_tools, "safe_retry_tool")
        object.__setattr__(self, "allowed_tools", allowed)
        object.__setattr__(self, "blocked_tools", blocked)
        object.__setattr__(self, "approval_tools", approvals)
        object.__setattr__(self, "safe_retry_tools", retryable)
        overlap = set(allowed) & set(blocked)
        if overlap:
            raise ValueError(
                "tools cannot be both allowed and blocked: "
                + ", ".join(sorted(overlap))
            )
        hidden_approvals = set(approvals) & set(blocked)
        if hidden_approvals:
            raise ValueError(
                "blocked tools cannot require approval: "
                + ", ".join(sorted(hidden_approvals))
            )

    def permits(self, tool_name: str) -> bool:
        """Return whether a tool is visible under this policy."""
        name = _required(tool_name, "tool_name")
        if name in self.blocked_tools:
            return False
        return not self.allowed_tools or name in self.allowed_tools

    def filter(
        self,
        tools: Sequence[MCPToolDescriptor],
    ) -> tuple[MCPToolDescriptor, ...]:
        """Filter tools deterministically while preserving server order."""
        return tuple(tool for tool in tools if self.permits(tool.name))

    def requires_approval(self, tool_name: str) -> bool:
        """Return whether a visible tool requires application approval."""
        name = _required(tool_name, "tool_name")
        return self.permits(name) and name in self.approval_tools

    def permits_retry(self, tool_name: str) -> bool:
        """Return whether a tool was explicitly classified safe for retry."""
        name = _required(tool_name, "tool_name")
        return self.permits(name) and name in self.safe_retry_tools


class MCPApprovalDecision(str, Enum):
    """Explicit application decision for one MCP tool call."""

    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class MCPApprovalRequest:
    """Approval request without logging sensitive arguments by representation."""

    server_label: str
    tool_name: str
    arguments: Mapping[str, Any] = field(repr=False)
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize identity and copy tool arguments."""
        object.__setattr__(
            self,
            "server_label",
            _required(self.server_label, "server_label"),
        )
        object.__setattr__(self, "tool_name", _required(self.tool_name, "tool_name"))
        object.__setattr__(self, "arguments", dict(self.arguments))


class MCPApprovalHandler(Protocol):
    """Application-owned approval callback."""

    def __call__(
        self,
        request: MCPApprovalRequest,
    ) -> MCPApprovalDecision | Awaitable[MCPApprovalDecision]:
        """Approve or reject one tool call explicitly."""
        ...


async def request_approval(
    policy: MCPToolPolicy,
    request: MCPApprovalRequest,
    handler: MCPApprovalHandler | None,
) -> MCPApprovalDecision:
    """Resolve explicit approval with fail-closed defaults.

    Tools not requiring approval are approved immediately. Tools requiring
    approval are rejected when no handler exists or when the handler returns an
    unsupported value.
    """
    if not policy.permits(request.tool_name):
        return MCPApprovalDecision.REJECT
    if not policy.requires_approval(request.tool_name):
        return MCPApprovalDecision.APPROVE
    if handler is None:
        return MCPApprovalDecision.REJECT
    decision = handler(request)
    if inspect.isawaitable(decision):
        decision = await cast(Awaitable[MCPApprovalDecision], decision)
    if not isinstance(decision, MCPApprovalDecision):
        return MCPApprovalDecision.REJECT
    return cast(MCPApprovalDecision, decision)


@dataclass(frozen=True, slots=True)
class MCPRetryPolicy:
    """Bounded retry policy restricted to explicit safe tool calls."""

    max_attempts: int = 1
    backoff_seconds: float = 0.0
    retryable_exceptions: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
    )

    def __post_init__(self) -> None:
        """Validate bounded retry configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        if not self.retryable_exceptions:
            raise ValueError("retryable_exceptions must not be empty")


async def run_safe_tool_call(
    tool_name: str,
    operation: Callable[[], T | Awaitable[T]],
    *,
    tool_policy: MCPToolPolicy,
    retry_policy: MCPRetryPolicy,
    idempotent: bool,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run one explicitly safe tool call with bounded retry.

    Retry attempts greater than one require both ``idempotent=True`` and the tool
    name in ``safe_retry_tools``. Mutating or unknown tools therefore fail before
    execution rather than being retried automatically.
    """
    name = _required(tool_name, "tool_name")
    if not tool_policy.permits(name):
        raise PermissionError(f"MCP tool {name!r} is not permitted")
    if retry_policy.max_attempts > 1 and (
        not idempotent or not tool_policy.permits_retry(name)
    ):
        raise ValueError(
            "Retries require idempotent=True and explicit safe_retry_tools policy"
        )
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            result = operation()
            if inspect.isawaitable(result):
                return await cast(Awaitable[T], result)
            return cast(T, result)
        except retry_policy.retryable_exceptions:
            if attempt >= retry_policy.max_attempts:
                raise
            if retry_policy.backoff_seconds:
                await sleep(retry_policy.backoff_seconds)
    raise RuntimeError("unreachable MCP retry state")


@dataclass(frozen=True, slots=True)
class MCPToolCacheEntry:
    """One immutable cached tool list and expiration timestamp."""

    tools: tuple[MCPToolDescriptor, ...]
    expires_at: float


class MCPToolCache:
    """Caller-owned in-memory tool-list cache with explicit invalidation."""

    def __init__(
        self,
        ttl_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[str, MCPToolCacheEntry] = {}

    def get(self, server_label: str) -> tuple[MCPToolDescriptor, ...] | None:
        """Return unexpired cached tools or remove an expired entry."""
        label = _required(server_label, "server_label")
        entry = self._entries.get(label)
        if entry is None:
            return None
        if entry.expires_at <= self._clock():
            self._entries.pop(label, None)
            return None
        return entry.tools

    def set(
        self,
        server_label: str,
        tools: Sequence[MCPToolDescriptor],
    ) -> tuple[MCPToolDescriptor, ...]:
        """Store a copied tool list and return the immutable value."""
        label = _required(server_label, "server_label")
        copied = tuple(tools)
        self._entries[label] = MCPToolCacheEntry(
            tools=copied,
            expires_at=self._clock() + self._ttl_seconds,
        )
        return copied

    def invalidate(self, server_label: str | None = None) -> None:
        """Invalidate one server entry or the entire cache explicitly."""
        if server_label is None:
            self._entries.clear()
            return
        self._entries.pop(_required(server_label, "server_label"), None)


@dataclass(frozen=True, slots=True)
class MCPServerFailure:
    """One isolated server tool-list failure with original exception."""

    server_label: str
    error: BaseException = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class MCPToolListReport:
    """Per-server filtered tool lists and isolated failures."""

    tools: Mapping[str, tuple[MCPToolDescriptor, ...]]
    failures: tuple[MCPServerFailure, ...]
    cache_hits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Copy report mappings and preserve deterministic order."""
        object.__setattr__(self, "tools", dict(self.tools))
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "cache_hits", tuple(self.cache_hits))

    @property
    def ok(self) -> bool:
        """Return whether every server listed tools successfully."""
        return not self.failures


async def list_tools_isolated(
    servers: Mapping[str, object],
    *,
    policy: MCPToolPolicy | None = None,
    cache: MCPToolCache | None = None,
) -> MCPToolListReport:
    """List and filter tools per server without one failure hiding others."""
    resolved_policy = policy or MCPToolPolicy()
    tool_map: dict[str, tuple[MCPToolDescriptor, ...]] = {}
    failures: list[MCPServerFailure] = []
    cache_hits: list[str] = []
    for server_label, server in servers.items():
        label = _required(server_label, "server_label")
        cached = cache.get(label) if cache is not None else None
        if cached is not None:
            tool_map[label] = resolved_policy.filter(cached)
            cache_hits.append(label)
            continue
        try:
            raw_tools = server.list_tools()  # type: ignore[attr-defined]
            if inspect.isawaitable(raw_tools):
                raw_tools = await cast(Awaitable[Sequence[object]], raw_tools)
            descriptors = tuple(_tool_descriptor(tool) for tool in raw_tools)
            if cache is not None:
                cache.set(label, descriptors)
            tool_map[label] = resolved_policy.filter(descriptors)
        except BaseException as error:
            failures.append(MCPServerFailure(label, error))
    return MCPToolListReport(tool_map, tuple(failures), tuple(cache_hits))


def build_agents_tool_filter(policy: MCPToolPolicy) -> object:
    """Build the official Agents SDK static MCP tool filter."""
    try:
        from agents.mcp import create_static_tool_filter
    except ImportError as error:
        raise ImportError(
            "MCP tool filtering requires a compatible openai-agents installation."
        ) from error
    return create_static_tool_filter(
        allowed_tool_names=(list(policy.allowed_tools) or None),
        blocked_tool_names=(list(policy.blocked_tools) or None),
    )


def _tool_descriptor(raw: object) -> MCPToolDescriptor:
    name = _read(raw, "name")
    if not name:
        raise ValueError("MCP tool is missing a name")
    description = _read(raw, "description")
    schema = _read(raw, "inputSchema")
    if schema is None:
        schema = _read(raw, "input_schema")
    if schema is not None and not isinstance(schema, Mapping):
        raise TypeError("MCP tool input schema must be a mapping")
    return MCPToolDescriptor(
        name=str(name),
        description=str(description) if description else None,
        input_schema=dict(schema) if isinstance(schema, Mapping) else None,
        raw=raw,
    )


def _read(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_required(value, name) for value in values))


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


__all__ = [
    "MCPApprovalDecision",
    "MCPApprovalHandler",
    "MCPApprovalRequest",
    "MCPRetryPolicy",
    "MCPServerFailure",
    "MCPToolCache",
    "MCPToolCacheEntry",
    "MCPToolDescriptor",
    "MCPToolListReport",
    "MCPToolPolicy",
    "build_agents_tool_filter",
    "list_tools_isolated",
    "request_approval",
    "run_safe_tool_call",
]

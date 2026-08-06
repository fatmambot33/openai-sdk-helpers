# MCP policy and resilience

MCP servers expose untrusted, changeable tools. Discovery confirms only that a
server returned a tool description; it does not establish safety, idempotence,
authorization, or approval.

`openai_sdk_helpers.mcp` therefore keeps policy explicit and caller-owned.
Nothing in this module executes a discovered tool automatically.

## Tool visibility

```python
from openai_sdk_helpers.mcp import MCPToolPolicy

policy = MCPToolPolicy(
    allowed_tools=("search", "read", "write"),
    blocked_tools=("write",),
    approval_tools=("read",),
    safe_retry_tools=("search",),
)
```

Rules are deterministic:

- a non-empty allow-list hides every unlisted tool;
- a block-list always wins;
- a tool cannot be both allowed and blocked;
- blocked tools cannot require approval;
- server order is preserved after filtering;
- unknown tools are hidden when an allow-list exists.

`build_agents_tool_filter(policy)` creates the official Agents SDK static MCP
tool filter. The official filter object can be passed to
`StreamableHTTPMCPConfig(tool_filter=...)`.

The package also exposes `MCPToolDescriptor` and `policy.filter()` for local
inspection and reporting. Filtering a tool list does not execute a tool or prove
that a server will continue exposing the same schema.

## Approval

Approval is application-owned and fail-closed:

```python
from openai_sdk_helpers.mcp import (
    MCPApprovalDecision,
    MCPApprovalRequest,
    request_approval,
)

request = MCPApprovalRequest(
    server_label="documents",
    tool_name="write",
    arguments={"path": "guide.md"},
)
decision = await request_approval(policy, request, approve_tool)
```

Behavior:

- blocked or hidden tools are rejected;
- visible tools not listed in `approval_tools` are approved by policy;
- tools requiring approval are rejected when no handler exists;
- unsupported handler values are rejected;
- sync and async handlers are accepted;
- arguments and raw tool-call objects are excluded from representation.

The package does not provide a user interface, approval queue, persistence
layer, or identity system. Applications must bind approval decisions to their
own authenticated user and authorization model.

Hosted MCP continues to use the official `require_approval` configuration. The
local approval helpers are for application-managed execution paths and policy
review; they do not override platform approval semantics silently.

## Tool-list caching

`MCPToolCache` is a caller-owned in-memory cache with a required positive TTL:

```python
from openai_sdk_helpers.mcp import MCPToolCache

cache = MCPToolCache(ttl_seconds=60)
```

The cache:

- is local to one object;
- stores immutable descriptor tuples;
- expires entries using a monotonic clock;
- removes expired entries on access;
- supports explicit per-server or global invalidation;
- does not persist across processes;
- does not refresh in the background;
- does not make a cached tool safe or available indefinitely.

For the official Agents SDK server cache, set
`StreamableHTTPMCPConfig(cache_tools_list=True)`. That setting is explicit and
independent from `MCPToolCache`.

Invalidate the application cache after deployments, authorization changes,
server version changes, or an unexpected tool-call failure suggesting stale
metadata.

## Isolated tool listing

```python
from openai_sdk_helpers.mcp import list_tools_isolated

report = await list_tools_isolated(
    {
        "documents": documents_server,
        "catalog": catalog_server,
    },
    policy=policy,
    cache=cache,
)
```

Each server is processed in mapping order. One list failure is stored as
`MCPServerFailure` and does not hide healthy servers. The report preserves:

- filtered descriptors by server label;
- original exceptions;
- cache-hit labels;
- deterministic ordering.

Malformed tool descriptions fail only their server entry. The helper does not
connect disconnected servers, retry listing, or execute tools.

## Bounded retries

Automatic retry of a mutating MCP tool can repeat side effects. Retries are
therefore denied unless all conditions are explicit:

1. `MCPRetryPolicy.max_attempts` is greater than one;
2. the tool is visible under policy;
3. the tool is listed in `safe_retry_tools`;
4. the call site passes `idempotent=True`;
5. the raised exception type is configured as retryable.

```python
from openai_sdk_helpers.mcp import MCPRetryPolicy, run_safe_tool_call

result = await run_safe_tool_call(
    "search",
    execute_search,
    tool_policy=policy,
    retry_policy=MCPRetryPolicy(
        max_attempts=3,
        backoff_seconds=0.5,
    ),
    idempotent=True,
)
```

Defaults allow one attempt only. The built-in retryable exceptions are
`TimeoutError` and `ConnectionError`. The original final exception is re-raised.
No jitter, exponential schedule, circuit breaker, or hidden retry is applied.
Applications needing those controls can supply a higher-level executor while
retaining the same explicit idempotence gate.

## Streamable HTTP integration

```python
tool_filter = build_agents_tool_filter(policy)
managed = build_streamable_http_server(
    StreamableHTTPMCPConfig(
        url="https://example.test/mcp",
        cache_tools_list=True,
        tool_filter=tool_filter,
    )
)
```

The cache and filter settings are forwarded to the official
`MCPServerStreamableHttp` constructor. The package does not configure official
SDK retry parameters; tool retries remain explicit through `run_safe_tool_call`.

## Failure isolation boundaries

Isolation prevents one tool-list failure from erasing healthy discovery results.
It does not:

- suppress connection or cleanup errors;
- convert failed tools into successful results;
- retry mutating operations;
- approve tools automatically;
- sandbox remote execution;
- guarantee server-side enforcement of a local policy;
- hide original official SDK objects or exceptions.

Applications should log safe identifiers and exception types, not tool
arguments, returned content, headers, authorization values, or full raw objects.

## Security review checklist

Before exposing an MCP server to a model:

- define an allow-list or justify an open tool set;
- block destructive or irrelevant tools;
- classify tools requiring approval;
- classify retry-safe tools separately;
- verify credentials and server URL ownership;
- decide cache TTL and invalidation events;
- test server-list and connection failures;
- ensure raw arguments and results are excluded from diagnostics;
- retain official tool-call and exception objects for audit;
- document cleanup and cancellation behavior.

A human review is required for changes to approval defaults, retry safety,
credential handling, tool filters, or trust boundaries.

# MCP transports

`openai_sdk_helpers.mcp` is an opt-in adapter layer over official OpenAI Agents
SDK and hosted MCP surfaces. It does not implement the Model Context Protocol,
discover servers globally, connect during import, execute tools during discovery,
or hide the official SDK objects.

Install the explicit profile before using this surface:

```console
pip install "openai-sdk-helpers[mcp]"
```

The base package already depends on the Agents SDK, which may itself carry a
compatible MCP dependency. The explicit `mcp` extra is nevertheless the stable
installation contract for applications that use this module. No MCP server is
constructed, connected, discovered, trusted, or exposed to a model by importing
the package.

## Supported transports

Issue #143 supports:

- the OpenAI platform-hosted MCP tool;
- the Agents SDK Streamable HTTP MCP server.

Stdio and legacy SSE are not exposed by this package. Add them only when a
concrete reusable workflow demonstrates a package adapter is clearer than direct
Agents SDK use.

## Hosted MCP

Hosted MCP is represented by `HostedMCPConfig`. The same explicit configuration
can be used directly with the Responses API or wrapped in the official Agents SDK
`HostedMCPTool`.

### Responses API

```python
from openai import OpenAI
from openai_sdk_helpers.mcp import HostedMCPConfig

client = OpenAI()
config = HostedMCPConfig(
    server_label="documentation",
    server_url="https://example.test/mcp",
    require_approval="always",
    allowed_tools=("search", "read"),
)

response = client.responses.create(
    model="gpt-5",
    input="Use the documentation server.",
    tools=[config.as_tool_config()],
)
```

### Agents SDK

```python
from agents import Agent
from openai_sdk_helpers.mcp import HostedMCPConfig, build_hosted_mcp_tool

config = HostedMCPConfig(
    server_label="documentation",
    server_url="https://example.test/mcp",
    require_approval="always",
    allowed_tools=("search", "read"),
)
tool = build_hosted_mcp_tool(config)
agent = Agent(name="Assistant", tools=[tool])
```

`as_tool_config()` is a local transformation. `build_hosted_mcp_tool()` performs
local official SDK object construction. Neither contacts the server nor runs a
model.

The configuration keeps these values explicit:

- server label and URL;
- approval policy;
- optional tool allow-list;
- optional server description;
- optional authorization value.

Authorization is excluded from dataclass representation. Applications remain
responsible for secret storage and must not place credentials in logs,
`OperationContext.metadata`, tests, or documentation.

Issue #144 adds reusable filtering and approval policy helpers. Until then,
`require_approval` and `allowed_tools` are explicit official SDK passthrough
values.

## Streamable HTTP

Create an unconnected official Agents SDK server:

```python
from agents import Agent, Runner
from openai_sdk_helpers.mcp import (
    StreamableHTTPMCPConfig,
    build_streamable_http_server,
)

managed = build_streamable_http_server(
    StreamableHTTPMCPConfig(
        url="https://example.test/mcp",
        headers={"Authorization": "Bearer <secret>"},
        timeout_seconds=10,
        sse_read_timeout_seconds=300,
        terminate_on_close=True,
        use_structured_content=True,
    )
)

assert not managed.connected
server = await managed.connect()
try:
    agent = Agent(name="Assistant", mcp_servers=[server])
    result = await Runner.run(agent, "Use the documentation server.")
finally:
    await managed.cleanup()
```

The wrapper forwards the URL, copied headers, request timeout, streaming read
timeout, termination behavior, optional display name, and structured-content
setting to the official `MCPServerStreamableHttp` constructor.

Construction does not connect. `connect()` and `cleanup()` are explicit and
idempotent at the wrapper level. The wrapper owns only the connection lifecycle
of the constructed SDK server. It does not own:

- remote server data;
- caller credentials;
- tool side effects;
- agent state;
- application retries;
- approval decisions.

The raw official server remains available as `raw_server` and is returned by
`connect()` and the async context manager.

## Async context lifecycle

```python
async with managed as server:
    agent = Agent(name="Assistant", mcp_servers=[server])
    ...
```

The context manager connects before returning the official server and cleans up
when leaving the block. A connection error is re-raised unchanged and does not
mark the wrapper connected. Cleanup after an unsuccessful connection is a no-op.

Cleanup errors are not suppressed. Applications decide whether a cleanup failure
is fatal, retried, or recorded as an operational incident.

## Observability

`connect()` and `cleanup()` accept an optional `OperationContext`:

```python
await managed.connect(
    operation_context=OperationContext(
        "mcp.documentation.connect",
        correlation_id="run-42",
        observers=(observe,),
    )
)
```

Safe diagnostics contain lifecycle metadata, not authorization headers, tool
arguments, tool results, prompts, or server content. Observers receive the raw
server or original exception in memory; applications must not export sensitive
objects accidentally.

Hosted tool construction is a local transformation and does not emit operation
events. Observe the surrounding Responses or Agents execution instead.

## Retries and failure behavior

Issue #143 adds no retry policy. Request timeouts are passed to the official
transport. Connection, protocol, authentication, tool-list, and cleanup failures
remain official SDK exceptions.

Issue #144 adds explicit bounded retry and failure-isolation policy where it can
be implemented without hiding SDK behavior. It must not retry mutating tool
calls automatically.

If the MCP/Agents transport integration cannot be imported, the builders fail
with the actionable installation command:

```console
pip install "openai-sdk-helpers[mcp]"
```

## Trust boundary

MCP discovery does not establish trust. A server can expose read-only or
mutating tools, change its tool list, return untrusted content, and perform remote
side effects. Applications must:

- configure an allow-list or filtering policy;
- require approval for sensitive tools;
- protect credentials;
- validate server URLs and deployment configuration;
- keep tool outputs out of logs by default;
- close transport connections explicitly;
- preserve official tool-call and exception objects for audit.

The package does not sandbox remote tools or guarantee that a tool marked
read-only is actually side-effect free.

## Compatibility

The MCP module is a focused import surface and is not added to the package root.
Existing Responses, Agents, Codex, retrieval, and direct official SDK workflows
are unchanged. The builders use lazy official SDK imports so simply importing
`openai_sdk_helpers.mcp` does not create a connection or perform discovery.

CI validates the explicit `mcp` install profile and the package's supported
minimum/latest OpenAI Agents dependency range. The Streamable HTTP adapter uses
stable official constructor fields shared by those supported versions; advanced
filtering, caching, retries, approval callbacks, and failure policies remain the
separate #144 layer.

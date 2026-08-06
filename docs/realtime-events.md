# Realtime events, tools, and deterministic testing

This guide covers the server-side event and control layer in
`openai_sdk_helpers.realtime`. It builds on the explicit session lifecycle in
[realtime.md](realtime.md) and keeps the official Agents SDK session available
for direct use.

## Design boundary

The helpers provide:

- ordered async event normalization with the original SDK event retained;
- sequential callback consumption using the same iterator;
- explicit tool parsing, application-owned registration, approval, timeout, and
  submission;
- explicit message, audio-chunk, interruption, and response-cancellation calls;
- a deterministic in-memory session for network-free tests.

They do not provide audio encoding, microphone or speaker management, browser
transport, automatic tool execution, hidden retries, a general event bus, or a
replacement for official SDK events.

## Ordered event consumption

```python
from openai_sdk_helpers.realtime import iter_realtime_events

async for event in iter_realtime_events(raw_session):
    print(event.kind, event.type)
    inspect_directly = event.raw
```

Every normalized event preserves its source object in `raw`. Source order is
preserved. In strict mode, malformed events raise
`RealtimeEventNormalizationError`. With `strict=False`, malformed events are
returned as `RealtimeEventKind.UNKNOWN` while retaining the raw object.

For callback-driven applications, use the same sequential state machine:

```python
from openai_sdk_helpers.realtime import consume_realtime_events

async def on_event(event):
    print(event.type)

count = await consume_realtime_events(raw_session, on_event)
```

Callbacks run one at a time in source order. A callback exception stops
consumption and is propagated unchanged. No background callback tasks are
created.

## Explicit tool execution

Tool handlers are registered locally and are never discovered or executed
implicitly:

```python
from openai_sdk_helpers.realtime import (
    RealtimeToolApprovalDecision,
    RealtimeToolRegistry,
    process_realtime_tool_event,
)

registry = RealtimeToolRegistry()
registry.register("lookup", lambda arguments: {"query": arguments["query"]})

async def approve(request):
    # The application owns identity, authorization, auditing, and UI policy.
    return RealtimeToolApprovalDecision.APPROVE

result = await process_realtime_tool_event(
    raw_session,
    registry,
    raw_tool_event,
    approval_handler=approve,
    timeout_seconds=10.0,
)
```

Approval is fail-closed: when approval is required and no valid approval callback
returns `APPROVE`, execution is rejected. Unknown tools raise `KeyError`. Tool
arguments are excluded from approval-request representations. Tool execution has
no hidden retry. Cancellation propagates, and an optional timeout cancels the
underlying handler task.

`execute_realtime_tool_call` executes locally but does not submit. The combined
`process_realtime_tool_event` helper parses, approves, executes, serializes, and
then explicitly invokes the session's tool-output method. Raw calls and original
exceptions remain accessible.

## Interruption and controls

```python
from openai_sdk_helpers.realtime import (
    cancel_realtime_response,
    interrupt_realtime_session,
    send_realtime_audio,
    send_realtime_message,
)

await send_realtime_message(raw_session, "Hello")
await send_realtime_audio(raw_session, audio_chunk)
await interrupt_realtime_session(raw_session)
await cancel_realtime_response(raw_session)
```

These helpers make one visible call to the supplied session. They do not manage
an audio buffer, retry, reconnect, or infer transport state. Empty messages and
audio chunks are rejected before the session call.

## Deterministic tests

`InMemoryRealtimeSession` is a recorded-operation fixture, not a protocol
emulator:

```python
from openai_sdk_helpers.realtime import InMemoryRealtimeSession

session = InMemoryRealtimeSession()
await session.push_event({"type": "session.created"})
await session.finish()

events = [event async for event in session]
assert events == [{"type": "session.created"}]
```

It records messages, copied audio chunks, tool outputs, interruption and
cancellation counts, and caller-pushed events. Closing is idempotent and finishes
event iteration. No network connection or credential is used.

## Failure and ownership rules

- The caller owns the official runner, session, transport, credentials, and
  connection policy.
- The application owns tool authorization and sensitive-action review.
- Approval defaults to rejection.
- Handler, callback, transport, timeout, and cancellation errors remain visible.
- No helper retries a tool or transport operation.
- `OperationContext` may observe explicit operations; official Agents SDK tracing
  remains authoritative for Realtime internals.
- Raw SDK events, sessions, calls, and exceptions remain available at every
  boundary.

## Compatibility

The module consumes structural event fields and session methods rather than
reconstructing the Realtime protocol. Direct official Agents SDK Realtime usage
and existing Responses websocket helpers remain supported. New SDK event types
are returned as `UNKNOWN` until a stable high-level classification is useful,
while the raw event remains intact.

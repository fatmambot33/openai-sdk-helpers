# Realtime API

`openai_sdk_helpers.realtime` provides a deliberately small server-side helper
surface over the official OpenAI Realtime SDK. It handles repeated client-secret
creation, connection-manager construction, session-update forwarding, shallow
event normalization, explicit response cancellation, and function-call output
submission.

It does not provide browser UI, audio capture or playback, WebRTC, a replacement
transport, reconnection, a second event protocol, or session persistence. Those
remain application responsibilities or official SDK surfaces.

## Compatibility and installation

The package-wide OpenAI Python SDK requirement is `openai>=3.41.0,<4.0.0`.
Realtime availability is checked structurally when a facade is constructed: the
injected client must expose `client.realtime.connect` and
`client.realtime.client_secrets.create`. An incompatible client raises
`RealtimeUnavailableError` before any network operation.

Opening a WebSocket connection additionally requires the official SDK's
Realtime extra:

```bash
pip install "openai[realtime]>=3.41.0,<4.0.0"
```

Creating a client secret is an HTTP request and uses the normal `openai`
installation. The returned value is a credential: pass it only to the intended
client over a secure channel; do not log, persist, or expose it in browser
source, diagnostics, or telemetry.

## Server-side session setup

`create_client_secret` forwards a native Realtime session mapping unchanged to
the official SDK. The original SDK response is returned.

```python
from openai import OpenAI
from openai_sdk_helpers.realtime import RealtimeClient

realtime = RealtimeClient(OpenAI())
session = realtime.create_client_secret(
    session={
        "type": "realtime",
        "model": "gpt-realtime",
        "output_modalities": ["text"],
        "instructions": "Answer clearly and briefly.",
    }
)

# Treat session.value as a credential.
```

Use the raw `realtime.client_secrets` resource whenever the installed SDK adds a new
session parameter before it earns a package helper.

## Connection lifecycle

`connect` forwards the model and low-level official SDK options and returns the
original connection manager. The caller owns the context manager and its
lifetime; the helper never opens, retains, reconnects, or closes a connection
behind the application's back.

```python
with realtime.connect(model="gpt-realtime") as connection:
    realtime.update_session(
        connection,
        session={"type": "realtime", "instructions": "Keep answers concise."},
    )
    for event in realtime.iter_events(connection):
        print(event.type)
        raw_event = event.raw
```

Use `async with` with `AsyncRealtimeClient.connect(...)`. The asynchronous
facade provides matching awaitable methods and an asynchronous `iter_events`.

## Event envelope

`normalize_realtime_event` returns a `RealtimeEvent` with a non-empty `type`
and the unchanged SDK event in `raw`. It accepts both official SDK event models
and mappings, which makes deterministic recorded-event tests possible without a
socket. It intentionally does not parse, replace, or persist event payloads.

```python
events = realtime.iter_events(connection)
first = next(events)
assert first.type == "session.updated"
assert first.raw.type == "session.updated"
```

## Cancellation and function tools

`cancel_response` makes cancellation explicit and forwards optional official
response and event identifiers. It does not clear audio output or emulate an
interruption workflow; applications using audio or WebRTC should use the raw SDK
resources and their relevant client-side lifecycle.

`submit_tool_result` creates one official `function_call_output` conversation
item and, by default, requests the next response. Set `create_response=False`
when the application must control response creation itself.

```python
realtime.submit_tool_result(
    connection,
    call_id="call_123",
    output='{"temperature": "18 C"}',
)
```

The helper accepts serialized `str` output rather than inventing a result schema
or silently serializing arbitrary application objects.

## Raw SDK escape hatches

Every facade exposes the unchanged official objects:

- `sdk_client` — configured OpenAI client;
- `realtime` — raw `client.realtime` resource;
- `client_secrets` — raw official Realtime client-secrets resource.

The connection manager returned by `connect`, its live connection, every raw
event, and upstream exceptions are preserved. Use these direct resources for
audio buffers, conversation editing, WebRTC, transcription sessions, new event
types, or other fast-moving Realtime behavior.

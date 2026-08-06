# Server-side Realtime sessions

`openai_sdk_helpers.realtime` provides typed configuration and explicit lifecycle
around the official OpenAI Agents SDK Realtime runner and session. It does not
implement the Realtime protocol, open a browser connection, capture audio,
manage microphones or speakers, create a WebRTC application, or reconnect
silently.

## Configuration

`RealtimeSessionConfig` represents server-side model and session settings:

```python
from openai_sdk_helpers.realtime import RealtimeSessionConfig

session_config = RealtimeSessionConfig(
    model="your-realtime-model",
    voice="your-voice",
    instructions="Answer clearly and briefly.",
    modalities=("audio", "text"),
    turn_detection={"type": "server_vad"},
    input_audio_transcription={"model": "your-transcription-model"},
)
```

Only explicitly configured values are serialized. Mutable mappings are copied,
modalities are deduplicated in caller order, and `extra` preserves official SDK
settings that do not yet justify a package field.

`as_model_config()` performs a local transformation only. It does not contact
OpenAI, load credentials, or create a session.

## Runner configuration

```python
from openai_sdk_helpers.realtime import RealtimeRunnerConfig

runner_config = RealtimeRunnerConfig(
    session=session_config,
    workflow_name="customer-support",
    group_id="conversation-42",
    trace_metadata={"tenant": "example"},
)
```

Runner configuration keeps official tracing controls visible:

- workflow name;
- group identifier;
- copied trace metadata;
- explicit tracing disablement;
- additional official runner settings through `extra`.

Do not put prompts, transcripts, authorization values, audio bytes, personal
data, or tool arguments in trace metadata.

## Build without connecting

```python
from openai_sdk_helpers.realtime import build_realtime_runner

runner = build_realtime_runner(
    starting_agent,
    config=runner_config,
)
```

The builder lazily imports and returns the official `RealtimeRunner`. It does not
run it. Applications may also construct a runner directly with the Agents SDK
and pass it to `manage_realtime_runner()`.

The package does not hide the official runner or session objects. They remain
available through `raw_runner`, `raw_session`, and the return value of `start()`.

## Explicit lifecycle

```python
from openai_sdk_helpers.realtime import manage_realtime_runner

managed = manage_realtime_runner(runner)
session = await managed.start()
try:
    # send input and consume events through the official session
    ...
finally:
    await managed.close()
```

Lifecycle states are:

- `created`;
- `starting`;
- `active`;
- `closing`;
- `closed`;
- `failed`.

Starting an active wrapper returns the same raw session. Closing before start or
closing more than once is safe and does not call the SDK twice. Connection and
close errors are re-raised unchanged.

## Async context management

```python
async with manage_realtime_runner(runner) as session:
    ...
```

The context manager starts the session, returns the original SDK session, and
closes it when leaving the block. It does not suppress close failures.

## Timeouts

```python
from openai_sdk_helpers.realtime import RealtimeLifecycleConfig

managed = manage_realtime_runner(
    runner,
    lifecycle=RealtimeLifecycleConfig(
        start_timeout_seconds=30,
        close_timeout_seconds=10,
    ),
)
```

The wrapper uses local async timeouts around session creation and close. A timeout
cancels the awaited operation, marks the wrapper failed, and raises
`asyncio.TimeoutError`. `None` disables the corresponding local timeout and
leaves timeout behavior to the official SDK and application.

No network retry or reconnect occurs automatically.

## Restart

Restart is disabled by default:

```python
lifecycle = RealtimeLifecycleConfig(allow_restart=True)
managed = manage_realtime_runner(runner, lifecycle=lifecycle)
new_session = await managed.restart()
```

An explicit restart closes the current session and calls the same runner again.
The application remains responsible for deciding whether state, tools, input
buffers, and conversation context should be reused. The package does not replay
audio, messages, or tool results.

## Observability

`start()` and `close()` accept an optional `OperationContext`:

```python
from openai_sdk_helpers import OperationContext

session = await managed.start(
    operation_context=OperationContext(
        "realtime.support.start",
        correlation_id="run-42",
        observers=(observe,),
    )
)
```

Events contain lifecycle metadata and the raw result or exception in memory.
Safe diagnostics must not export transcripts, audio, model events, tool
arguments, authorization headers, or raw session objects by default.

Official Agents SDK tracing remains the source of truth for agent, tool, and
Realtime execution traces. `OperationContext` complements that tracing with a
small application lifecycle hook; it does not replace or duplicate the SDK trace
model.

## Cancellation

Task cancellation propagates normally. The wrapper does not convert
`CancelledError` into success or continue work in the background. A cancelled
start or close marks the wrapper failed. The application decides whether to
perform a later explicit close or restart.

## Ownership

The wrapper owns only the start/close lifecycle it was asked to manage. It does
not own:

- credentials or API clients;
- remote conversation resources;
- application message history;
- audio capture, playback, or buffers;
- tool side effects;
- approval decisions;
- retry policy;
- browser or mobile transports.

Closing a local wrapper does not delete server-side conversations or application
state unless the official session itself documents that behavior.

## Compatibility

The Realtime module is a focused import surface and is not added to the package
root. Existing Responses websocket helpers and direct Agents SDK Realtime usage
remain unchanged.

Session construction is lazy so importing the package does not require or start a
Realtime connection. An actionable import error is raised only when the builder
is invoked with an Agents SDK version that lacks the required Realtime runner.

Issue #146 adds normalized event envelopes, explicit tool execution,
interruption, cancellation helpers, and a deterministic test transport on top of
this lifecycle contract.

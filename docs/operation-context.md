# Operation context and observability

`OperationContext` is an optional, per-operation contract shared by the package's
Responses runner, Agents runner, and Codex command registry. It carries caller
metadata and emits lifecycle events without installing a telemetry backend or
replacing OpenAI Agents SDK tracing.

## Basic use

```python
from openai_sdk_helpers import OperationContext, OperationEvent
from openai_sdk_helpers.response import run_sync


def observe(event: OperationEvent) -> None:
    print(event.diagnostics())


context = OperationContext(
    "responses.review",
    correlation_id="job-42",
    trace_id="caller-trace-7",
    metadata={"tenant": "example"},
    observers=(observe,),
)

result = run_sync(
    ReviewResponse,
    content="Review this code.",
    operation_context=context,
)
```

Passing no context preserves existing behavior. The runner returns the original
result and re-raises the original exception.

## Lifecycle

One context represents exactly one operation and emits these phases:

1. `OperationPhase.START`
2. `OperationPhase.SUCCESS`, or
3. `OperationPhase.FAILURE`

Start, success, and failure observers run inline. An observer failure increments
`observer_error_count` and does not replace the operation result or exception.
Create a fresh context for every concurrent operation. A context is predictably
mutable during its lifecycle and must not be shared concurrently. Observers that
are shared across threads or tasks are responsible for their own synchronization.

Sync and async runner paths emit equivalent phases. `CodexPluginRegistry.run()`
continues to return an awaitable for async commands; completion is emitted when
that awaitable is resolved. `run_async()` observes the full resolved command.

## Metadata and usage

A context can carry:

- request, correlation, and trace identifiers;
- an operation name;
- model metadata;
- copied caller metadata;
- an explicit retry count;
- optional token usage.

On successful completion, common SDK-shaped `model`, `request_id`, `_request_id`,
and `usage` fields are captured when present. Usage fields remain optional and
never change the returned SDK object.

Call `record_retry()` only when the caller or wrapper knows that a retry occurred.
The runtime does not infer retries or add its own retry policy.

## Safe diagnostics

`OperationEvent.diagnostics()` produces a JSON-compatible mapping and excludes:

- the raw result;
- exception messages;
- prompts, input, output, response content, files, and tool arguments;
- credentials and authorization-like metadata.

Redaction is case-insensitive and configurable with `redact_keys`. Setting
`include_sensitive=True` permits metadata values not covered by caller-provided
redaction keys, but raw results and exception messages are still never
serialized.

Observers receive the original `result` or `error` object on the in-memory event
for integrations that require direct access. Callers are responsible for not
exporting sensitive objects.

## Relationship to official SDK features

This contract is deliberately vendor-neutral glue:

- Agents SDK tracing remains the source of truth for agent spans and trace
  configuration.
- Responses identifiers and raw results remain available unchanged.
- Codex commands keep their original return and exception behavior.
- No global context, exporter, background worker, or telemetry dependency is
  installed.

Use official SDK tracing or direct SDK result objects whenever they already meet
the workflow's needs. Use `OperationContext` only for shared application metadata
and lifecycle observation across package surfaces.

## Compatibility

The new `operation_context` keyword is optional on:

- `openai_sdk_helpers.response.run_sync` and `run_async`;
- `openai_sdk_helpers.agent.run_sync` and `run_async`;
- `CodexPluginRegistry.run` and `run_async`.

Existing calls require no migration. The operation runtime is also available
directly through `run_observed_sync` and `run_observed_async` for application
operations that do not use a package runner.

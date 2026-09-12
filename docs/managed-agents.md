# Managed Agents API

`openai_sdk_helpers.managed_agents` provides a thin, typed facade over the
OpenAI-managed Agents API introduced in the OpenAI Python SDK 3.13.0.

This surface is distinct from `openai_sdk_helpers.agent`, which composes the
application-run OpenAI Agents SDK. Managed Agents sessions are durable OpenAI API
resources whose orchestration, context management, environment, artifacts, and
subagents are owned by the official API.

## Compatibility

The package-wide OpenAI Python SDK constraint remains `>=2.45.0,<4.0.0` so
existing Responses, Agents SDK, retrieval, files, and other helpers continue to
work on the supported minimum dependency set.

Managed Agents helpers specifically require an SDK client exposing
`client.beta.agents.sessions`, first shipped in `openai` 3.13.0. Importing the
module remains safe on older supported SDK versions. Constructing a managed
Agents facade against an incompatible client raises
`ManagedAgentsUnavailableError` before any API call with the installed SDK
version and required capability in the error context.

Applications can check explicitly:

```python
from openai import OpenAI

from openai_sdk_helpers.managed_agents import managed_agents_available

client = OpenAI()
if managed_agents_available(client):
    ...
```

For an environment pinned below 3.13.0, upgrade only when this feature is needed:

```bash
pip install "openai>=3.13.0,<4.0.0"
```

## Session lifecycle

The facade covers repeated discrete session plumbing and returns the original
SDK objects unchanged:

```python
from openai import OpenAI

from openai_sdk_helpers.managed_agents import ManagedAgentsClient

client = OpenAI()
managed = ManagedAgentsClient(client)

session = managed.create_session(
    environment={"type": "computer", "name": "default"},
    agent_id="agent_123",
    input="Inspect the repository and summarize the changes.",
)

same_session = managed.retrieve_session(session.id)
page = managed.list_sessions(agent_id="agent_123", limit=20)
managed.update_session_metadata(session.id, metadata={"workflow": "review"})
managed.delete_session(session.id)
```

`AsyncManagedAgentsClient` provides matching asynchronous methods over an
`AsyncOpenAI` client.

## Continuing a session

`submit_events` forwards official message, cancellation, or tool-result input
events and returns the SDK result unchanged:

```python
managed.submit_events(
    session.id,
    events=[
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Continue."}],
        }
    ],
    idempotency_key="continue-review-1",
)
```

The helper does not invent a second conversation store or orchestration layer.
Session state stays owned by the managed Agents API.

## Raw resources and streaming

The underlying SDK surfaces are intentional escape hatches:

```python
raw_agents = managed.agents
raw_sessions = managed.sessions
raw_client = managed.sdk_client
```

Use the official SDK resource directly for streaming and fast-moving beta
surfaces:

```python
with managed.sessions.stream(
    session.id,
    input="Run the tests and report failures.",
) as stream:
    for event in stream:
        print(event)

artifacts = managed.sessions.artifacts.list(session.id)
subagents = managed.sessions.subagents.list(session.id)
```

This package deliberately does not normalize the SDK event stream, artifact
model, sandbox/environment model, subagent model, or tool-handling protocol.
Those capabilities remain directly accessible as OpenAI evolves them.

## Observability

Discrete facade requests accept the package's optional `OperationContext` and
reuse `run_observed_sync` / `run_observed_async`. The original SDK result or
exception is preserved. Streaming is not wrapped by `OperationContext`, because
its lifecycle is already represented by the official stream object and events.

Safe diagnostics continue to redact content-bearing metadata by default; callers
that inspect raw managed-agent events or artifacts are responsible for handling
their content appropriately.

## Choosing between execution surfaces

Use **Responses** when the application wants direct request/response control and
owns orchestration. Use the **Agents SDK** when the application runs the agent
loop locally and benefits from SDK tools, handoffs, guardrails, sessions, and
tracing. Use the **Managed Agents API** when the application needs OpenAI-managed
durable agent sessions and the managed execution harness.

When a helper would only rename an official Agents API method, prefer
`managed.sessions`, `managed.agents`, or the original OpenAI client directly.

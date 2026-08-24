# Conversation state and persistence

Choose one state owner for each workflow. The package does not automatically
select a memory mechanism, merge incompatible histories, create a database, or
hide official OpenAI identifiers and session objects.

## State modes

| Mode | Owner | Public contract | Lifecycle |
| --- | --- | --- | --- |
| Stateless | Caller | No state object or an empty `AgentRunState` / `ResponseContinuation` | Each call is independent |
| Application history | Caller | Caller-owned input items, result input lists, or `ResponseMessages` | Caller decides what to retain and resend |
| Package-local persistence | Caller and local filesystem | `LocalMessageStore` with `ResponseMessages` | Explicit `save`, `resume`, `clear`, and `delete` |
| Agents session | Official Agents SDK session implementation | `AgentRunState(session=...)` or existing `session=...` shorthand | Session owns retrieval, insertion, clearing, and storage |
| Previous response | OpenAI server | `previous_response_id`; Agents may also use `auto_previous_response_id` | Caller retains response identifiers and decides when to continue |
| Conversation | OpenAI server | `conversation_id` for Agents; `conversation` request parameter for Responses | Caller retains the conversation identifier and server resource ownership |

## Compatibility matrix

The package validates these combinations before invoking an SDK runner:

| Session | Previous response | Auto previous response | Conversation | Supported |
| --- | --- | --- | --- | --- |
| No | No | No | No | Yes — stateless |
| Yes | No | No | No | Yes — Agents SDK session |
| No | Yes | No or Yes | No | Yes — server response chain |
| No | No | No | Yes | Yes — server conversation |
| Yes | Any | Any | Any | No |
| No | Yes | Any | Yes | No |
| No | No | Yes | Yes | No |

A `ResponseContinuation` accepts either `previous_response_id` or
`conversation_id`, never both. Its `apply()` method copies official Responses
request keyword arguments and adds the corresponding `previous_response_id` or
`conversation` value without hiding the raw identifier.

An `AgentRunState` follows the official Agents SDK rules:

- a session cannot be combined with `conversation_id`,
  `previous_response_id`, or `auto_previous_response_id`;
- a conversation cannot be combined with either previous-response option;
- an explicit `previous_response_id` may be paired with
  `auto_previous_response_id=True` when the SDK should continue chaining after
  the first turn.

## Agents runner usage

Existing stateless calls are unchanged:

```python
result = run_sync(agent, "Hello")
```

The existing session shorthand remains supported:

```python
result = run_sync(agent, "Hello", session=session)
```

Use `state` when choosing server-managed continuation explicitly:

```python
from openai_sdk_helpers import AgentRunState
from openai_sdk_helpers.agent import run_sync

state = AgentRunState(previous_response_id="resp_123")
result = run_sync(agent, "Continue", state=state)
```

The runner forwards the underlying session or identifiers to the official SDK.
It returns the original result and preserves access to fields such as
`last_response_id`. Passing both `state` and `session` fails before the API call.
Sync and async runner functions use the same validation and forwarding rules.

## Responses continuation

`ResponseContinuation` is a small request adapter for direct Responses SDK calls
or custom package wrappers:

```python
from openai_sdk_helpers import ResponseContinuation

continuation = ResponseContinuation(previous_response_id="resp_123")
request = continuation.apply(
    {
        "model": "your-model",
        "input": "Continue the answer.",
    }
)
response = client.responses.create(**request)
```

For a server conversation:

```python
continuation = ResponseContinuation(conversation_id="conv_123")
request = continuation.apply({"model": "your-model", "input": "Next turn"})
```

The adapter does not create, retrieve, clear, or delete server resources. Those
operations remain explicit official SDK calls owned by the application.

## Application-owned history

The package's existing Responses workflows maintain `ResponseMessages` for
building request payloads. This is application-owned history, not an Agents SDK
session or server conversation. Callers may inspect and serialize the collection
and remain responsible for what is included in future requests.

Do not combine a complete application-managed transcript with a server-managed
continuation identifier unless the specific official SDK workflow explicitly
requires both. Duplicate context can increase cost and produce unexpected model
behavior. The package does not silently merge them.

## Package-local persistence

`LocalMessageStore` makes filesystem ownership explicit:

```python
from openai_sdk_helpers import LocalMessageStore, ResponseMessages

store = LocalMessageStore("state/review.json")
messages = ResponseMessages()
store.save(messages)
restored = store.resume()
```

Lifecycle behavior:

- construction and `close(save=False)` perform no write;
- `save(messages)` creates parent directories and overwrites the configured JSON
  file;
- `resume()` reads that file and raises `FileNotFoundError` when absent;
- `clear()` replaces saved history with an empty `ResponseMessages` collection;
- `delete()` removes the caller-owned file and reports whether it existed;
- `close(messages, save=True)` is an explicit save-on-close operation.

No background autosave, lock manager, database, encryption layer, retention
policy, or remote synchronization is introduced. Sensitive message storage,
filesystem permissions, encryption, backups, and deletion policy remain caller
responsibilities.

## Existing `ResponseBase` persistence

`ResponseBase` retains its existing `save_messages`, `save()`, and `close()`
behavior for compatibility. Existing applications do not need an immediate
migration. New reusable workflows should prefer an explicit state owner and use
`LocalMessageStore` when local persistence is required.

A future breaking release may separate `ResponseBase` resource cleanup from its
legacy autosave behavior, but no deprecation is introduced in this release.

## Cleanup boundaries

- Closing a package runner cleans up only resources already documented as owned
  by that runner.
- Closing a `LocalMessageStore` does not delete or save unless explicitly asked.
- Clearing or deleting an Agents SDK session uses methods on the underlying
  session object.
- Clearing or deleting an OpenAI conversation is an explicit official SDK
  resource operation.
- A response or conversation identifier is never deleted merely because a local
  object is closed.

## Migration guidance

Most callers require no change:

- stateless calls remain stateless;
- existing `session=session` calls remain supported;
- direct official SDK parameters remain available;
- existing `ResponseBase` local persistence behavior remains compatible.

Adopt `AgentRunState` when a workflow needs to document its ownership mode or
use server-managed continuation. Adopt `ResponseContinuation` for direct
Responses request construction. Adopt `LocalMessageStore` when save/resume and
cleanup must be explicit and independently testable.

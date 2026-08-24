# Retrieval lifecycle clients

`OpenAIRetrievalClient` and `AsyncOpenAIRetrievalClient` implement explicit Files
and Vector Stores lifecycle operations over caller-injected official SDK clients.
They do not construct clients, own credentials, close injected clients, close
caller-owned file handles, or delete remote resources automatically.

The lifecycle-only structural contracts are
`SyncRetrievalLifecycleClient` and `AsyncRetrievalLifecycleClient`. The broader
`SyncRetrievalClient` and `AsyncRetrievalClient` contracts also include direct
search and are completed by the File Search phase in issue #142.

## Construction

```python
from openai import OpenAI
from openai_sdk_helpers.retrieval import OpenAIRetrievalClient

sdk_client = OpenAI()
retrieval = OpenAIRetrievalClient(sdk_client)

assert retrieval.sdk_client is sdk_client
```

Use `AsyncOpenAIRetrievalClient` with an injected `AsyncOpenAI` client. The sync
and async clients expose matching lifecycle semantics.

## Files

Upload one file explicitly:

```python
result = retrieval.upload_file("guide.pdf", purpose="user_data")
file = result.resource

assert result.succeeded
assert file is not None
print(file.id)
print(file.raw)
```

The source may be a path, string path, or caller-owned binary file object. The
wrapper passes the source to the official SDK and never closes it. Optional file
expiration is forwarded only when supplied.

Batch upload preserves input order:

```python
batch = retrieval.upload_files(
    ("one.pdf", "two.pdf"),
    purpose="user_data",
    continue_on_error=True,
)

for outcome in batch.results:
    if outcome.succeeded:
        print(outcome.resource.id)
    else:
        print(type(outcome.error).__name__)
```

With `continue_on_error=True`, ordinary per-item exceptions retain the original
exception and later inputs continue. With `False`, the original exception is
raised immediately. Task cancellation and process-level interrupts are never
converted into item failures: `asyncio.CancelledError`, `KeyboardInterrupt`, and
`SystemExit` propagate. Uploads are sequential and make no hidden concurrency or
retry promise.

Deleting a Files resource is always separate and explicit:

```python
retrieval.delete_file(file.id)
```

## Vector stores

Create a vector store with visible lifecycle settings:

```python
store_result = retrieval.create_vector_store(
    name="product-guides",
    file_ids=(file.id,),
    metadata={"team": "docs"},
    expires_after={"anchor": "last_active_at", "days": 7},
    chunking_strategy={"type": "auto"},
)
store = store_result.resource
```

The client also exposes one-page retrieval, listing, and update operations:

```python
current = retrieval.retrieve_vector_store(store.id)
stores = retrieval.list_vector_stores(limit=20, order="desc")
updated = retrieval.update_vector_store(store.id, name="guides-v2")
```

List results preserve official SDK order. Name lookup is intentionally absent;
resource IDs are the stable identity.

Deletion is explicit:

```python
retrieval.delete_vector_store(store.id)
```

Deleting a vector store does not imply deletion of its underlying Files
resources.

## Attach, upload, and polling

Attach an existing Files resource and poll through the official SDK helper:

```python
from openai_sdk_helpers.retrieval import PollingConfig

attachment = retrieval.attach_file(
    store.id,
    file.id,
    attributes={"region": "eu"},
    polling=PollingConfig(
        poll_interval_ms=500,
        timeout_seconds=30,
    ),
)
```

For `attach_file`, `PollingConfig` forwards `poll_interval_ms` and the optional
request timeout to the official SDK `create_and_poll` helper. The timeout controls
the SDK request that creates the attachment; it is not an overall ingestion
deadline. Configured terminal states default to `completed`, `failed`, and
`cancelled`. If the helper returns a different status, the wrapper raises
`RuntimeError` rather than claiming ingestion completed. Only `completed` is
reported with `succeeded=True`; failed and cancelled terminal resources remain
available with `succeeded=False` and their raw SDK state intact.

Upload directly through the vector-store helper when a separate Files upload is
not required:

```python
attachment = retrieval.upload_and_poll(
    store.id,
    "guide.pdf",
    polling=PollingConfig(poll_interval_ms=500),
)
```

The official SDK `upload_and_poll` helper supports the polling interval but does
not expose a request-timeout keyword. Supplying `timeout_seconds` to this package
path therefore raises `ValueError` before any API call. When request-timeout
control is required, upload explicitly with `upload_file` and then call
`attach_file` with the returned file ID.

Both attachment methods expose the raw SDK vector-store file object and its
status through `VectorStoreFileReference`.

## Detach versus delete

Detaching removes membership from one vector store:

```python
retrieval.detach_file(store.id, file.id)
```

It does not delete the underlying Files resource. Delete that resource only with
an explicit `delete_file(file.id)` call.

This distinction is preserved in operation names, result types, tests, and
migration guidance.

## Ownership and cleanup

The lifecycle clients track no hidden ownership set. Construction and ordinary
method completion perform no cleanup. There is intentionally no cleanup-on-close
or cleanup-on-context-exit behavior.

A temporary workflow should record successful results and explicitly delete only
resources it created:

```python
store_result = retrieval.create_vector_store(name="temporary")
try:
    # use the store
    ...
finally:
    if store_result.resource is not None:
        retrieval.delete_vector_store(store_result.resource.id)
```

Applications must preserve partial cleanup failures and must not delete external
IDs merely because they were passed to the wrapper.

## Observability

Every network operation accepts an optional `OperationContext`:

```python
from openai_sdk_helpers import OperationContext

context = OperationContext(
    "retrieval.files.upload",
    correlation_id="job-42",
    observers=(observe,),
)
result = retrieval.upload_file(
    "guide.pdf",
    purpose="user_data",
    operation_context=context,
)
```

Observers receive lifecycle events and the original normalized result. Safe
diagnostics exclude file bytes, query content, filters, attributes, and raw SDK
objects unless an application deliberately exports them. Observer failures do
not replace SDK results or exceptions.

## Compatibility

Existing `FilesAPIManager`, `VectorStorage`, response file processing, Agents
message builders, and response vector-store helpers remain available. The new
clients do not emit deprecation warnings because #141 does not yet provide every
File Search adapter required for a complete migration.

Issue #142 adds direct vector-store search normalization and File Search request
adapters. Deprecation begins only after equivalent replacements and migration
tests exist.

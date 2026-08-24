# Retrieval architecture and migration map

The target retrieval package is `openai_sdk_helpers.retrieval`. It uses official
OpenAI terminology and separates three concerns:

1. Files API resource lifecycle;
2. vector-store lifecycle and direct search;
3. File Search tool configuration and application message composition.

Issue #140 approves contracts and migration boundaries only. Network operations
are implemented in #141 and File Search request/result adapters in #142.

## Design principles

- The caller injects an official `OpenAI` or `AsyncOpenAI` client.
- The helper never silently constructs a client from an API key.
- Files and vector stores remain distinct resources.
- Attaching a file to a vector store does not transfer ownership of the Files
  resource.
- Detaching a vector-store file does not delete the underlying Files resource.
- File and vector-store deletion are separate, explicit operations.
- Store creation, chunking, attributes, expiration, filters, ranking, query
  rewriting, and result limits remain visible.
- Every normalized resource or result preserves the original SDK object through
  `raw`.
- Sync and async implementations share models but use separate protocols.
- Application message construction remains outside resource lifecycle clients.
- No non-OpenAI vector database abstraction is introduced.

## Target public surface

The architecture contract exports from `openai_sdk_helpers.retrieval`:

- `UploadedFile`
- `VectorStoreReference`
- `RetrievalOperationResult`
- `RetrievalSearchContent`
- `RetrievalSearchResult`
- `RetrievalSearchPage`
- `FileSearchConfig`
- `SyncRetrievalClient`
- `AsyncRetrievalClient`
- `FileSource`
- `AttributeValue`

These names are intentionally not package-root exports during preview. Import
from the focused module so the 0.9 review can evolve the surface without
expanding the already-large package root.

## Resource models

### Uploaded files

`UploadedFile` normalizes the official file identifier, filename, purpose, and
optional byte count. The underlying SDK `FileObject` remains available as
`raw`.

Uploading a file does not imply:

- vector-store creation;
- attachment to a vector store;
- automatic cleanup;
- expiration unless explicitly requested;
- ownership transfer to a response or agent wrapper.

### Vector stores

`VectorStoreReference` holds a store ID, optional name, and the raw official SDK
resource. Name lookup is not part of the identity contract because names are not
unique resource identifiers.

### Operation outcomes

`RetrievalOperationResult[T]` records the operation name, normalized resource,
success flag, raw SDK response, and optional original exception for explicit
batch or cleanup workflows. Ordinary single-resource methods may still raise the
original SDK exception. Implementations must document which behavior they use.

### Search results

`RetrievalSearchResult` normalizes source file identity, relevance score,
attributes, and ordered text fragments while preserving the raw result item.
`RetrievalSearchPage` preserves query strings, order, pagination state, and the
raw SDK page.

Normalization does not synthesize citations, summarize content, or discard
attributes. Applications retain full control over presentation and prompting.

## File Search configuration

`FileSearchConfig` represents only the official File Search tool settings:

- vector-store IDs;
- maximum result count;
- attribute filters;
- ranking options;
- whether Responses should include raw File Search results.

`as_tool()` creates an SDK-shaped tool dictionary. `response_includes()` returns
`file_search_call.results` only when explicitly requested. Store creation,
upload, and search execution are separate operations.

## Client protocols

`SyncRetrievalClient` and `AsyncRetrievalClient` define matching operations:

- upload a file;
- create a vector store;
- attach an existing file;
- detach a file without deleting it;
- search a vector store;
- delete an underlying file;
- delete a vector store;
- expose the injected official SDK client.

The protocols deliberately do not define:

- implicit context-manager deletion;
- global registries;
- automatic name lookup;
- automatic upload when configuring File Search;
- automatic deletion after a response;
- a proprietary filter language;
- a generic third-party vector database API.

## Existing-surface inventory

| Existing surface | Current behavior | Decision | Migration boundary |
| --- | --- | --- | --- |
| `FilesAPIManager` | Files CRUD, batch upload, tracking, context-manager cleanup, default `user_data` expiration | **Adapt and deprecate gradually** | Implement the new file operations with injected client and explicit ownership; retain legacy class as an adapter for at least one minor line |
| `FilePurpose` | Literal purpose alias | **Keep temporarily** | Re-export or alias from the implementation until official SDK typing can be used directly without compatibility loss |
| `VectorStorage` | Store creation/discovery, uploads, downloads, deletion, cleanup, implicit ownership | **Adapt** | Use as implementation input, then route public lifecycle through the new client while preserving existing imports |
| `VectorStorageFileInfo` / `VectorStorageFileStats` | Batch status and error summaries | **Deprecate after adapters ship** | Map to `RetrievalOperationResult` collections; no immediate removal |
| `vector_storage.cleanup` | Best-effort tracked cleanup | **Internalize** | Keep explicit cleanup utilities behind the implementation; never make cleanup implicit in the new protocol |
| `response.files.process_files` | Inline encoding plus hidden vector-store creation/upload and response-owned cleanup | **Split** | Keep inline image/PDF composition; route vector lifecycle through an injected retrieval client and require explicit ownership |
| `response.vector_store.attach_vector_store` | Resolves non-unique names, may construct a client, mutates protected response tool state | **Deprecate** | Replace with explicit store IDs and `FileSearchConfig`; keep a compatibility adapter during 0.9 |
| `agent.files.build_agent_input_messages` | Uploads documents while composing input messages | **Keep and adapt** | Accept an injected retrieval uploader; message construction remains in the Agents surface |
| `agent.search.vector` | Agent orchestration over File Search | **Keep** | Consume `FileSearchConfig` and retrieval references without owning remote resource lifecycle |
| `structure.vector_search` | Application-oriented plan/report schemas | **Keep** | Not a resource or raw search-result model; remains separate |
| `response` and `agent` private vector fields | Hidden store ownership and cleanup | **Internalize then remove privately** | Preserve behavior until migration tests exist; new code must use explicit lifecycle contracts |

## Compatibility and deprecation plan

### 0.9 preview

- Add the retrieval contracts and implementations.
- Keep every existing import working.
- Add adapters from legacy classes and helpers.
- Emit deprecation warnings only where a complete replacement exists.
- Document destructive and network behavior at each call site.

### Following minor line

- Prefer `openai_sdk_helpers.retrieval` in examples and documentation.
- Stop expanding legacy managers.
- Keep compatibility adapters and migration tests.
- Review whether package-root legacy exports remain justified.

### Future major release

Removal is considered only after:

- at least one documented deprecation window;
- equivalent sync/async functionality;
- migration guidance for ownership and cleanup;
- evidence that public consumers can preserve raw SDK access;
- explicit human approval.

## Ownership examples

### Caller-owned resources

The default new-client contract treats uploaded files and vector stores as
caller-owned. Closing the client does not delete them.

```python
file_result = retrieval.upload_file("guide.pdf", purpose="user_data")
store_result = retrieval.create_vector_store(
    name="guides",
    file_ids=(file_result.resource.id,),
)
```

Deletion remains two explicit actions when both resources should be removed:

```python
retrieval.delete_vector_store(store_result.resource.id)
retrieval.delete_file(file_result.resource.id)
```

### Temporary workflows

Applications may build their own scoped cleanup policy from returned operation
results. The package may offer an explicit owned-resource scope in a later issue,
but it must list resources before deletion, preserve partial failures, and never
claim ownership of externally supplied IDs.

## Security and diagnostics

Files and search content may be sensitive. Implementations must:

- avoid logging file bytes, extracted text, query content, filters, or attributes
  by default;
- integrate optional `OperationContext` metadata without exporting content;
- preserve original SDK exceptions;
- make destructive operations explicit;
- document retention and expiration parameters;
- use synthetic data and mocked SDK clients in pull-request tests.

## Approval boundary

This document and the contract tests define the proposed 0.9 public surface.
Because #140 is an architecture and public-API decision, the pull request remains
draft until a human reviewer approves naming, ownership, and migration choices.
Implementation issues #141 and #142 must not broaden the approved surface without
updating this document and receiving equivalent review.

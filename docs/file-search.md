# Direct vector-store search and hosted File Search

The retrieval package exposes two related but distinct workflows:

1. **direct vector-store search** through `OpenAIRetrievalClient.search()` or
   `AsyncOpenAIRetrievalClient.search()`;
2. **model-assisted File Search** configured as a hosted tool for Responses or
   the Agents SDK.

They share result models and filter vocabulary, but they do not share execution
semantics. Direct search returns scored chunks immediately. Hosted File Search
runs inside a model request and may expose tool-call results and file citations.

## Direct vector-store search

```python
from openai import OpenAI
from openai_sdk_helpers.retrieval import OpenAIRetrievalClient

retrieval = OpenAIRetrievalClient(OpenAI())
page = retrieval.search(
    "vs_123",
    "Which warranty applies in Europe?",
    max_num_results=10,
    ranking_options={"score_threshold": 0.35},
    rewrite_query=True,
)

for result in page.data:
    print(result.file_id, result.score)
    for fragment in result.content:
        print(fragment.text)
```

The method forwards:

- one query string or an ordered list of query strings;
- optional attribute filters;
- `max_num_results` from 1 through 50;
- ranking options;
- optional query rewriting.

The page preserves SDK order, pagination information, query values, attributes,
content fragments, and the raw SDK page and result objects.

## Attribute filters

Use validated filter builders for common official operators:

```python
from openai_sdk_helpers.retrieval import (
    ComparisonFilter,
    ComparisonOperator,
    CompoundFilter,
    CompoundOperator,
)

region = ComparisonFilter(
    key="region",
    operator=ComparisonOperator.EQ,
    value="eu",
)
year = ComparisonFilter(
    key="year",
    operator=ComparisonOperator.IN,
    value=(2025, 2026),
)
filters = CompoundFilter(
    operator=CompoundOperator.AND,
    filters=(region, year),
)

page = retrieval.search("vs_123", "warranty", filters=filters)
```

Supported comparison operators are `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`,
and `nin`. `in` and `nin` require a non-empty tuple. Compound groups require at
least two child filters and support `and` and `or`.

Applications may pass an explicit official SDK filter mapping instead of a
builder. `serialize_filter()` copies passthrough mappings and does not invent a
separate filter language.

## Result normalization

`normalize_search_page()` and the client search methods operate in strict mode by
default. A malformed item raises `RetrievalNormalizationError` with its result
index and preserves the original exception as the cause.

```python
page = retrieval.search("vs_123", "warranty", strict=False)
```

Lenient mode omits malformed items, retains valid items in API order, and keeps
the raw SDK page available for inspection. It does not replace missing content,
synthesize scores, or convert malformed attributes silently.

An empty SDK result list is a valid `RetrievalSearchPage(data=())`.

## Responses File Search

Create one explicit configuration:

```python
from openai_sdk_helpers.retrieval import (
    FileSearchConfig,
    apply_file_search_to_response,
)

config = FileSearchConfig(
    vector_store_ids=("vs_123",),
    max_num_results=8,
    filters=filters.as_dict(),
    ranking_options={"score_threshold": 0.35},
    include_search_results=True,
)

request = apply_file_search_to_response(
    {
        "model": "your-model",
        "input": "Summarize the warranty rules.",
    },
    config,
)
response = client.responses.create(**request)
```

`apply_file_search_to_response()` copies the request mapping, appends an official
SDK-shaped `file_search` tool, and adds `file_search_call.results` to `include`
only when requested. Existing tools and include values retain their order and the
input mapping is not mutated.

The adapter does not upload files, create stores, make a model request, or own
remote resources.

## Agents File Search

Convert the same configuration to the official Agents SDK hosted tool:

```python
from openai_sdk_helpers.retrieval import build_agents_file_search_tool

file_search_tool = build_agents_file_search_tool(config)
```

Only configured optional values are passed to `FileSearchTool`, preserving
compatibility with the supported minimum Agents SDK. The returned object is the
official SDK tool and remains fully accessible to the caller.

## Included tool-call results

When Responses includes File Search results, normalize one tool call explicitly:

```python
from openai_sdk_helpers.retrieval import normalize_file_search_call

call = normalize_file_search_call(raw_file_search_call)
for result in call.results:
    print(result.filename, result.score)
```

`FileSearchCall` preserves:

- the tool call ID;
- generated or used queries;
- status;
- included normalized results in API order;
- the original SDK call object.

Strict and lenient malformed-result handling matches direct search.

If results were not included, `results` is empty; the adapter does not issue a
second request or claim the model used a particular source.

## Citations

Model output can contain file citation annotations. Collect them in output order:

```python
from openai_sdk_helpers.retrieval import collect_file_citations

for citation in collect_file_citations(response):
    print(citation.file_id, citation.filename, citation.index)
```

`FileCitation` preserves the raw annotation. Citation extraction does not fetch
file contents, verify claims, format footnotes, or deduplicate repeated
annotations.

## Observability and sensitive data

Direct search accepts an optional `OperationContext`. Safe diagnostics must not
export query text, filter values, attributes, result content, or raw SDK objects
by default. Applications that inspect event results are responsible for content
handling and retention.

Hosted-tool adapters are local transformations and do not emit lifecycle events.
Observe the surrounding Responses or Agents run instead.

## Ownership and cleanup

Search and tool configuration do not transfer resource ownership:

- vector stores remain caller-owned;
- attached Files resources remain caller-owned;
- direct search does not create or delete resources;
- a Responses or Agents run does not delete a store when it finishes;
- citation extraction does not retrieve or retain source file bytes.

Use the explicit lifecycle methods documented in
[retrieval-lifecycle.md](retrieval-lifecycle.md) for attachment, detachment, and
deletion.

## Legacy compatibility

Existing response vector-store helpers, `VectorStorage`, `FilesAPIManager`, and
Agents file-message builders remain available. The new retrieval package is the
preferred surface for new code, but no existing import is removed in 0.9.

A deprecation warning is introduced only after an equivalent adapter exists and
its ownership behavior is covered by migration tests. The migration map remains
canonical in [retrieval.md](retrieval.md).

"""Tests for direct vector-store search and hosted File Search adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openai_sdk_helpers.retrieval import (
    AsyncOpenAIRetrievalClient,
    ComparisonFilter,
    ComparisonOperator,
    CompoundFilter,
    CompoundOperator,
    FileSearchConfig,
    OpenAIRetrievalClient,
    RetrievalNormalizationError,
    apply_file_search_to_response,
    build_agents_file_search_tool,
    collect_file_citations,
    normalize_file_search_call,
    normalize_search_page,
    serialize_filter,
)


def _valid_result(file_id: str = "file_1", score: float = 0.9) -> object:
    return SimpleNamespace(
        file_id=file_id,
        filename=f"{file_id}.txt",
        score=score,
        attributes={"region": "eu"},
        content=(SimpleNamespace(type="text", text=f"content for {file_id}"),),
    )


class SyncVectorStores:
    """SDK-shaped vector-store resource with direct search."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.page = SimpleNamespace(
            data=(_valid_result(),),
            has_more=True,
            next_page="cursor_2",
        )

    def search(self, vector_store_id: str, **kwargs: Any) -> object:
        self.calls.append((vector_store_id, kwargs))
        return self.page


class SyncClient:
    """Minimal synchronous SDK client."""

    def __init__(self) -> None:
        self.files = object()
        self.vector_stores = SyncVectorStores()


class AsyncVectorStores:
    """SDK-shaped asynchronous vector-store resource with direct search."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search(self, vector_store_id: str, **kwargs: Any) -> object:
        self.calls.append((vector_store_id, kwargs))
        return SimpleNamespace(data=(_valid_result("file_async"),), has_more=False)


class AsyncClient:
    """Minimal asynchronous SDK client."""

    def __init__(self) -> None:
        self.files = object()
        self.vector_stores = AsyncVectorStores()


def test_comparison_and_compound_filters_serialize_official_shape() -> None:
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
    combined = CompoundFilter(
        operator=CompoundOperator.AND,
        filters=(region, year),
    )

    assert region.as_dict() == {
        "type": "eq",
        "key": "region",
        "value": "eu",
    }
    assert combined.as_dict() == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "region", "value": "eu"},
            {"type": "in", "key": "year", "value": [2025, 2026]},
        ],
    }
    passthrough = {"type": "eq", "key": "team", "value": "docs"}
    copied = serialize_filter(passthrough)
    assert copied == passthrough
    assert copied is not passthrough


@pytest.mark.parametrize(
    "filter_value",
    [
        ComparisonFilter,
        lambda: ComparisonFilter("region", ComparisonOperator.IN, ()),
        lambda: ComparisonFilter("region", ComparisonOperator.EQ, ("eu",)),
        lambda: CompoundFilter(
            CompoundOperator.AND,
            (ComparisonFilter("region", ComparisonOperator.EQ, "eu"),),
        ),
    ],
)
def test_filter_validation_rejects_invalid_shapes(filter_value: object) -> None:
    if filter_value is ComparisonFilter:
        with pytest.raises(TypeError):
            ComparisonFilter()  # type: ignore[call-arg]
        return
    with pytest.raises((TypeError, ValueError)):
        filter_value()  # type: ignore[operator]


def test_sync_search_forwards_explicit_settings_and_normalizes_page() -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)
    search_filter = ComparisonFilter(
        key="region",
        operator=ComparisonOperator.EQ,
        value="eu",
    )

    page = client.search(
        "vs_1",
        ("first", "second"),
        filters=search_filter,
        max_num_results=12,
        ranking_options={"score_threshold": 0.4},
        rewrite_query=True,
    )

    assert page.query == ("first", "second")
    assert page.has_more is True
    assert page.next_page == "cursor_2"
    assert page.data[0].file_id == "file_1"
    assert page.data[0].raw is sdk.vector_stores.page.data[0]
    assert sdk.vector_stores.calls == [
        (
            "vs_1",
            {
                "query": ["first", "second"],
                "filters": {"type": "eq", "key": "region", "value": "eu"},
                "max_num_results": 12,
                "ranking_options": {"score_threshold": 0.4},
                "rewrite_query": True,
            },
        )
    ]


def test_sync_search_forwards_only_normalized_query_values() -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)

    page = client.search("vs_1", (" first ", " ", "second"))

    assert page.query == ("first", "second")
    assert sdk.vector_stores.calls[0][1]["query"] == ["first", "second"]


@pytest.mark.parametrize("value", [0, 51])
def test_search_rejects_invalid_result_limits_before_api_call(value: int) -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)

    with pytest.raises(ValueError, match="between 1 and 50"):
        client.search("vs_1", "question", max_num_results=value)

    assert sdk.vector_stores.calls == []


def test_sync_search_rejects_more_than_five_normalized_queries() -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)

    with pytest.raises(ValueError, match="at most 5"):
        client.search("vs_1", ("one", "two", "three", "four", "five", "six"))

    assert sdk.vector_stores.calls == []


@pytest.mark.asyncio
async def test_async_search_matches_sync_result_contract() -> None:
    sdk = AsyncClient()
    client = AsyncOpenAIRetrievalClient(sdk)

    page = await client.search("vs_async", "question")

    assert page.query == ("question",)
    assert page.data[0].file_id == "file_async"
    assert sdk.vector_stores.calls == [("vs_async", {"query": "question"})]


@pytest.mark.asyncio
async def test_async_search_rejects_more_than_five_normalized_queries() -> None:
    sdk = AsyncClient()
    client = AsyncOpenAIRetrievalClient(sdk)

    with pytest.raises(ValueError, match="at most 5"):
        await client.search(
            "vs_async",
            ("one", "two", "three", "four", "five", "six"),
        )

    assert sdk.vector_stores.calls == []


def test_empty_search_results_are_valid() -> None:
    raw = SimpleNamespace(data=(), has_more=False)

    page = normalize_search_page(raw, query="nothing")

    assert page.data == ()
    assert page.raw is raw


def test_malformed_results_support_strict_and_lenient_modes() -> None:
    valid = _valid_result()
    malformed = SimpleNamespace(file_id="file_bad", score="high", content=())
    raw = SimpleNamespace(data=(valid, malformed), has_more=False)

    with pytest.raises(RetrievalNormalizationError) as exc_info:
        normalize_search_page(raw, query="question", strict=True)
    assert exc_info.value.index == 1

    page = normalize_search_page(raw, query="question", strict=False)
    assert [result.file_id for result in page.data] == ["file_1"]
    assert page.raw is raw


def test_missing_filename_is_malformed_in_strict_and_lenient_modes() -> None:
    missing_filename = SimpleNamespace(
        file_id="file_missing_name",
        score=0.8,
        attributes={},
        content=(SimpleNamespace(type="text", text="content"),),
    )
    raw = SimpleNamespace(data=(missing_filename,), has_more=False)

    with pytest.raises(RetrievalNormalizationError, match="filename"):
        normalize_search_page(raw, query="question", strict=True)

    page = normalize_search_page(raw, query="question", strict=False)
    assert page.data == ()


@pytest.mark.parametrize(
    "attributes",
    [
        {1: "eu"},
        {"tags": ["a"]},
    ],
)
def test_malformed_attributes_are_rejected_or_omitted(attributes: object) -> None:
    malformed = SimpleNamespace(
        file_id="file_bad_attributes",
        filename="bad.txt",
        score=0.8,
        attributes=attributes,
        content=(SimpleNamespace(type="text", text="content"),),
    )
    raw = SimpleNamespace(data=(malformed,), has_more=False)

    with pytest.raises(RetrievalNormalizationError, match="attribute"):
        normalize_search_page(raw, query="question", strict=True)

    page = normalize_search_page(raw, query="question", strict=False)
    assert page.data == ()


def test_responses_adapter_copies_and_appends_without_mutation() -> None:
    config = FileSearchConfig(
        vector_store_ids=("vs_1",),
        max_num_results=4,
        include_search_results=True,
    )
    original = {
        "model": "example-model",
        "tools": [{"type": "web_search"}],
        "include": ["reasoning.encrypted_content"],
    }

    resolved = apply_file_search_to_response(original, config)

    assert original["tools"] == [{"type": "web_search"}]
    assert original["include"] == ["reasoning.encrypted_content"]
    assert resolved["tools"] == [
        {"type": "web_search"},
        {
            "type": "file_search",
            "vector_store_ids": ["vs_1"],
            "max_num_results": 4,
        },
    ]
    assert resolved["include"] == [
        "reasoning.encrypted_content",
        "file_search_call.results",
    ]


def test_agents_adapter_passes_only_configured_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeFileSearchTool:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    import agents

    monkeypatch.setattr(agents, "FileSearchTool", FakeFileSearchTool)
    config = FileSearchConfig(
        vector_store_ids=("vs_1",),
        include_search_results=False,
    )

    tool = build_agents_file_search_tool(config)

    assert isinstance(tool, FakeFileSearchTool)
    assert captured == {
        "vector_store_ids": ["vs_1"],
        "include_search_results": False,
    }


def test_file_search_call_and_citations_preserve_raw_objects() -> None:
    raw_result = _valid_result()
    raw_call = SimpleNamespace(
        id="fs_1",
        queries=("first", "second"),
        status="completed",
        results=(raw_result,),
    )
    call = normalize_file_search_call(raw_call)

    annotation = SimpleNamespace(
        type="file_citation",
        file_id="file_1",
        filename="guide.pdf",
        index=3,
    )
    response = SimpleNamespace(
        output=(
            SimpleNamespace(
                content=(SimpleNamespace(annotations=(annotation,)),),
            ),
        )
    )
    citations = collect_file_citations(response)

    assert call.raw is raw_call
    assert call.queries == ("first", "second")
    assert call.results[0].raw is raw_result
    assert citations[0].file_id == "file_1"
    assert citations[0].filename == "guide.pdf"
    assert citations[0].raw is annotation

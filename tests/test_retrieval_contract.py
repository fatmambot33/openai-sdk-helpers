"""Contract tests for the approved retrieval target surface."""

from __future__ import annotations

from openai_sdk_helpers import retrieval
from openai_sdk_helpers.retrieval import (
    FileSearchConfig,
    RetrievalOperationResult,
    RetrievalSearchContent,
    RetrievalSearchPage,
    RetrievalSearchResult,
    UploadedFile,
    VectorStoreReference,
)

EXPECTED_RETRIEVAL_API = (
    "AsyncRetrievalClient",
    "AttributeValue",
    "FileSearchConfig",
    "FileSource",
    "RetrievalOperationResult",
    "RetrievalSearchContent",
    "RetrievalSearchPage",
    "RetrievalSearchResult",
    "SyncRetrievalClient",
    "UploadedFile",
    "VectorStoreReference",
)


def test_retrieval_module_exports_are_explicit() -> None:
    assert tuple(retrieval.__all__) == EXPECTED_RETRIEVAL_API
    assert all(hasattr(retrieval, name) for name in EXPECTED_RETRIEVAL_API)


def test_resource_types_preserve_raw_sdk_objects() -> None:
    raw_file = object()
    raw_store = object()
    uploaded = UploadedFile(
        id="file_123",
        filename="guide.pdf",
        purpose="user_data",
        bytes=42,
        raw=raw_file,
    )
    store = VectorStoreReference(id="vs_123", name="guides", raw=raw_store)

    assert uploaded.raw is raw_file
    assert store.raw is raw_store
    assert RetrievalOperationResult(
        operation="files.upload",
        resource=uploaded,
        succeeded=True,
        raw=raw_file,
    ).raw is raw_file


def test_file_search_config_is_explicit_and_deduplicated() -> None:
    config = FileSearchConfig(
        vector_store_ids=("vs_1", "vs_1", "vs_2"),
        max_num_results=5,
        filters={"type": "eq", "key": "region", "value": "eu"},
        ranking_options={"score_threshold": 0.4},
        include_search_results=True,
    )

    assert config.vector_store_ids == ("vs_1", "vs_2")
    assert config.as_tool() == {
        "type": "file_search",
        "vector_store_ids": ["vs_1", "vs_2"],
        "max_num_results": 5,
        "filters": {"type": "eq", "key": "region", "value": "eu"},
        "ranking_options": {"score_threshold": 0.4},
    }
    assert config.response_includes() == ("file_search_call.results",)


def test_search_results_are_normalized_without_losing_raw_access() -> None:
    raw_item = object()
    raw_page = object()
    item = RetrievalSearchResult(
        file_id="file_123",
        filename="guide.pdf",
        score=0.91,
        content=(RetrievalSearchContent("Relevant paragraph"),),
        attributes={"region": "eu", "year": 2026},
        raw=raw_item,
    )
    page = RetrievalSearchPage(
        query=("first question", "second question"),
        data=(item,),
        has_more=True,
        next_page="cursor_123",
        raw=raw_page,
    )

    assert page.data[0].raw is raw_item
    assert page.raw is raw_page
    assert page.query == ("first question", "second question")
    assert page.data[0].content[0].text == "Relevant paragraph"

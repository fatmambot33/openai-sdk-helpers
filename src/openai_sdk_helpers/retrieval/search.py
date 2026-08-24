"""Direct vector-store search and hosted File Search adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias, cast

from openai_sdk_helpers.runtime import (
    OperationContext,
    run_observed_async,
    run_observed_sync,
)

from .contracts import (
    AttributeValue,
    FileSearchConfig,
    RetrievalSearchContent,
    RetrievalSearchPage,
    RetrievalSearchResult,
)

FilterValue: TypeAlias = AttributeValue | tuple[AttributeValue, ...]


class ComparisonOperator(str, Enum):
    """Official vector-store attribute comparison operators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NIN = "nin"


class CompoundOperator(str, Enum):
    """Official compound attribute filter operators."""

    AND = "and"
    OR = "or"


class SearchFilter(Protocol):
    """Serializable official vector-store attribute filter."""

    def as_dict(self) -> dict[str, Any]:
        """Return the SDK-shaped filter mapping."""
        ...


@dataclass(frozen=True, slots=True)
class ComparisonFilter:
    """Validated common attribute comparison filter.

    Parameters
    ----------
    key : str
        File attribute key.
    operator : ComparisonOperator
        Official comparison operator.
    value : AttributeValue or tuple[AttributeValue, ...]
        Scalar value for ordinary comparisons or a non-empty tuple for
        ``in``/``nin``.
    """

    key: str
    operator: ComparisonOperator
    value: FilterValue

    def __post_init__(self) -> None:
        """Normalize the key and validate scalar versus collection values."""
        key = self.key.strip()
        if not key:
            raise ValueError("key must not be empty")
        object.__setattr__(self, "key", key)
        collection_operator = self.operator in {
            ComparisonOperator.IN,
            ComparisonOperator.NIN,
        }
        if collection_operator:
            if not isinstance(self.value, tuple) or not self.value:
                raise ValueError("in and nin filters require a non-empty tuple")
        elif isinstance(self.value, tuple):
            raise ValueError("only in and nin filters accept tuple values")

    def as_dict(self) -> dict[str, Any]:
        """Return the SDK-shaped comparison filter."""
        value: AttributeValue | list[AttributeValue]
        if isinstance(self.value, tuple):
            value = list(self.value)
        else:
            value = self.value
        return {
            "type": self.operator.value,
            "key": self.key,
            "value": value,
        }


@dataclass(frozen=True, slots=True)
class CompoundFilter:
    """Validated ``and`` or ``or`` group of attribute filters."""

    operator: CompoundOperator
    filters: tuple[SearchFilter, ...]

    def __post_init__(self) -> None:
        """Require at least two serializable child filters."""
        filters = tuple(self.filters)
        if len(filters) < 2:
            raise ValueError("compound filters require at least two children")
        for child in filters:
            if not hasattr(child, "as_dict"):
                raise TypeError("compound filter children must define as_dict()")
        object.__setattr__(self, "filters", filters)

    def as_dict(self) -> dict[str, Any]:
        """Return the SDK-shaped compound filter."""
        return {
            "type": self.operator.value,
            "filters": [child.as_dict() for child in self.filters],
        }


@dataclass(frozen=True, slots=True)
class FileSearchCall:
    """Normalized model-assisted File Search call.

    Parameters
    ----------
    id : str
        Tool call identifier.
    queries : tuple[str, ...]
        Queries generated or used by the model.
    status : str
        Tool call status.
    results : tuple[RetrievalSearchResult, ...]
        Included results in API order. Empty when results were not requested.
    raw : object or None, default=None
        Original SDK tool call object.
    """

    id: str
    queries: tuple[str, ...]
    status: str
    results: tuple[RetrievalSearchResult, ...] = ()
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize identifiers, status, queries, and result order."""
        call_id = self.id.strip()
        status = self.status.strip()
        queries = tuple(query.strip() for query in self.queries if query.strip())
        if not call_id:
            raise ValueError("id must not be empty")
        if not status:
            raise ValueError("status must not be empty")
        object.__setattr__(self, "id", call_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "results", tuple(self.results))


@dataclass(frozen=True, slots=True)
class FileCitation:
    """Normalized file citation from model output annotations."""

    file_id: str
    filename: str | None = None
    index: int | None = None
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize citation identity and optional metadata."""
        file_id = self.file_id.strip()
        if not file_id:
            raise ValueError("file_id must not be empty")
        object.__setattr__(self, "file_id", file_id)
        if self.filename is not None:
            filename = self.filename.strip()
            object.__setattr__(self, "filename", filename or None)
        if self.index is not None and self.index < 0:
            raise ValueError("index must be non-negative")


class RetrievalNormalizationError(ValueError):
    """Malformed SDK search data that cannot be normalized strictly."""

    def __init__(self, index: int, message: str) -> None:
        super().__init__(f"Search result {index}: {message}")
        self.index = index


def serialize_filter(
    value: SearchFilter | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Serialize a validated filter or copy an SDK passthrough mapping.

    Parameters
    ----------
    value : SearchFilter, Mapping[str, Any], or None
        Validated filter builder or explicit official SDK mapping.

    Returns
    -------
    dict[str, Any] or None
        Copied SDK-shaped filter.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return value.as_dict()


def normalize_search_page(
    raw_page: object,
    *,
    query: str | Sequence[str],
    strict: bool = True,
) -> RetrievalSearchPage:
    """Normalize one direct vector-store search page.

    Parameters
    ----------
    raw_page : object
        Official SDK search page.
    query : str or Sequence[str]
        Query supplied to the search operation.
    strict : bool, default=True
        Raise on malformed result items. When disabled, malformed items are
        omitted while valid items retain API order and ``raw_page`` remains
        available for inspection.

    Returns
    -------
    RetrievalSearchPage
        Normalized page. Empty SDK data produces ``data=()``.
    """
    queries = _queries(query)
    normalized: list[RetrievalSearchResult] = []
    for index, raw_item in enumerate(_read(raw_page, "data", ()) or ()):
        try:
            normalized.append(_normalize_result(raw_item))
        except (TypeError, ValueError) as error:
            if strict:
                raise RetrievalNormalizationError(index, str(error)) from error
    next_page = _read(raw_page, "next_page")
    return RetrievalSearchPage(
        query=queries,
        data=tuple(normalized),
        has_more=bool(_read(raw_page, "has_more", False)),
        next_page=str(next_page) if next_page else None,
        raw=raw_page,
    )


def normalize_file_search_call(
    raw_call: object,
    *,
    strict: bool = True,
) -> FileSearchCall:
    """Normalize one Responses File Search tool call and included results."""
    raw_results = _read(raw_call, "results", ()) or ()
    results: list[RetrievalSearchResult] = []
    for index, raw_item in enumerate(raw_results):
        try:
            results.append(_normalize_result(raw_item))
        except (TypeError, ValueError) as error:
            if strict:
                raise RetrievalNormalizationError(index, str(error)) from error
    call_id = _read(raw_call, "id")
    status = _read(raw_call, "status")
    return FileSearchCall(
        id=str(call_id or "unknown-file-search-call"),
        queries=_queries(_read(raw_call, "queries", ()) or ()),
        status=str(status or "unknown"),
        results=tuple(results),
        raw=raw_call,
    )


def collect_file_citations(response: object) -> tuple[FileCitation, ...]:
    """Collect file citation annotations from an SDK response in output order."""
    citations: list[FileCitation] = []
    for output in _read(response, "output", ()) or ():
        for content in _read(output, "content", ()) or ():
            for annotation in _read(content, "annotations", ()) or ():
                if _read(annotation, "type") != "file_citation":
                    continue
                file_id = _read(annotation, "file_id")
                if not file_id:
                    continue
                filename = _read(annotation, "filename")
                index = _read(annotation, "index")
                citations.append(
                    FileCitation(
                        file_id=str(file_id),
                        filename=str(filename) if filename else None,
                        index=index if isinstance(index, int) else None,
                        raw=annotation,
                    )
                )
    return tuple(citations)


def apply_file_search_to_response(
    request_kwargs: Mapping[str, Any],
    config: FileSearchConfig,
) -> dict[str, Any]:
    """Copy Responses request kwargs and append explicit File Search settings."""
    resolved = dict(request_kwargs)
    tools: list[Any] = list(cast(Sequence[Any], resolved.get("tools") or ()))
    tools.append(config.as_tool())
    resolved["tools"] = tools
    includes: list[str] = [
        str(value) for value in cast(Sequence[Any], resolved.get("include") or ())
    ]
    for include in config.response_includes():
        if include not in includes:
            includes.append(include)
    if includes:
        resolved["include"] = includes
    return resolved


def build_agents_file_search_tool(config: FileSearchConfig) -> object:
    """Create the official Agents SDK ``FileSearchTool`` from one config."""
    from agents import FileSearchTool

    kwargs: dict[str, Any] = {
        "vector_store_ids": list(config.vector_store_ids),
        "include_search_results": config.include_search_results,
    }
    if config.max_num_results is not None:
        kwargs["max_num_results"] = config.max_num_results
    if config.ranking_options is not None:
        kwargs["ranking_options"] = dict(config.ranking_options)
    if config.filters is not None:
        kwargs["filters"] = dict(config.filters)
    return FileSearchTool(**kwargs)


class SyncSearchMixin:
    """Direct vector-store search for a synchronous retrieval client."""

    def search(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        *,
        filters: SearchFilter | Mapping[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: Mapping[str, Any] | None = None,
        rewrite_query: bool | None = None,
        strict: bool = True,
        operation_context: OperationContext | None = None,
    ) -> RetrievalSearchPage:
        """Search one vector store and normalize results in SDK order."""
        vector_store_id = _identifier(vector_store_id, "vector_store_id")
        queries = _queries(query)
        _validate_max_results(max_num_results)
        sdk_client = cast(Any, self).sdk_client

        def execute() -> RetrievalSearchPage:
            kwargs = _search_kwargs(
                query=query,
                filters=filters,
                max_num_results=max_num_results,
                ranking_options=ranking_options,
                rewrite_query=rewrite_query,
            )
            raw = sdk_client.vector_stores.search(vector_store_id, **kwargs)
            return normalize_search_page(raw, query=queries, strict=strict)

        return run_observed_sync(operation_context, execute)


class AsyncSearchMixin:
    """Direct vector-store search for an asynchronous retrieval client."""

    async def search(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        *,
        filters: SearchFilter | Mapping[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: Mapping[str, Any] | None = None,
        rewrite_query: bool | None = None,
        strict: bool = True,
        operation_context: OperationContext | None = None,
    ) -> RetrievalSearchPage:
        """Search one vector store and normalize results in SDK order."""
        vector_store_id = _identifier(vector_store_id, "vector_store_id")
        queries = _queries(query)
        _validate_max_results(max_num_results)
        sdk_client = cast(Any, self).sdk_client

        async def execute() -> RetrievalSearchPage:
            kwargs = _search_kwargs(
                query=query,
                filters=filters,
                max_num_results=max_num_results,
                ranking_options=ranking_options,
                rewrite_query=rewrite_query,
            )
            raw = await sdk_client.vector_stores.search(vector_store_id, **kwargs)
            return normalize_search_page(raw, query=queries, strict=strict)

        return await run_observed_async(operation_context, execute)


def _normalize_result(raw_item: object) -> RetrievalSearchResult:
    file_id = _read(raw_item, "file_id") or _read(raw_item, "id")
    filename = _read(raw_item, "filename") or file_id
    score = _read(raw_item, "score")
    if not file_id:
        raise ValueError("file_id is missing")
    if not filename:
        raise ValueError("filename is missing")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise TypeError("score must be numeric")
    if not 0 <= float(score) <= 1:
        raise ValueError("score must be between 0 and 1")
    content = _content(raw_item)
    attributes = _read(raw_item, "attributes", {}) or {}
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping")
    return RetrievalSearchResult(
        file_id=str(file_id),
        filename=str(filename),
        score=float(score),
        content=content,
        attributes=dict(attributes),
        raw=raw_item,
    )


def _content(raw_item: object) -> tuple[RetrievalSearchContent, ...]:
    raw_content = _read(raw_item, "content")
    if raw_content is None:
        text = _read(raw_item, "text")
        raw_content = ({"type": "text", "text": text},) if text else ()
    content: list[RetrievalSearchContent] = []
    for item in raw_content or ():
        text = _read(item, "text")
        content_type = _read(item, "type", "text")
        if not text:
            continue
        content.append(
            RetrievalSearchContent(
                text=str(text),
                type=str(content_type or "text"),
            )
        )
    if not content:
        raise ValueError("content is missing")
    return tuple(content)


def _queries(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        values = ()
    queries = tuple(str(item).strip() for item in values if str(item).strip())
    if not queries:
        raise ValueError("query must not be empty")
    return queries


def _search_kwargs(
    *,
    query: str | Sequence[str],
    filters: SearchFilter | Mapping[str, Any] | None,
    max_num_results: int | None,
    ranking_options: Mapping[str, Any] | None,
    rewrite_query: bool | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "query": query if isinstance(query, str) else list(query),
    }
    serialized_filter = serialize_filter(filters)
    if serialized_filter is not None:
        kwargs["filters"] = serialized_filter
    if max_num_results is not None:
        kwargs["max_num_results"] = max_num_results
    if ranking_options is not None:
        kwargs["ranking_options"] = dict(ranking_options)
    if rewrite_query is not None:
        kwargs["rewrite_query"] = rewrite_query
    return kwargs


def _validate_max_results(value: int | None) -> None:
    if value is not None and not 1 <= value <= 50:
        raise ValueError("max_num_results must be between 1 and 50")


def _identifier(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _read(value: object, name: str, default: object | None = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "AsyncSearchMixin",
    "ComparisonFilter",
    "ComparisonOperator",
    "CompoundFilter",
    "CompoundOperator",
    "FileCitation",
    "FileSearchCall",
    "RetrievalNormalizationError",
    "SearchFilter",
    "SyncSearchMixin",
    "apply_file_search_to_response",
    "build_agents_file_search_tool",
    "collect_file_citations",
    "normalize_file_search_call",
    "normalize_search_page",
    "serialize_filter",
]

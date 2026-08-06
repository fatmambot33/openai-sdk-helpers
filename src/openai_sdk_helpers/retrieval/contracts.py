"""Public contracts for OpenAI file, vector-store, and search workflows."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Generic, Protocol, TypeVar, runtime_checkable

ResourceT = TypeVar("ResourceT")
FileSource = BinaryIO | Path | str
AttributeValue = str | bool | int | float


def _identifier(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """Uploaded OpenAI file with direct access to the SDK resource.

    Parameters
    ----------
    id : str
        OpenAI file identifier.
    filename : str
        Server-visible filename.
    purpose : str
        Official Files API purpose.
    bytes : int or None, default=None
        File size reported by the API.
    raw : object or None, default=None
        Underlying official SDK file resource.
    """

    id: str
    filename: str
    purpose: str
    bytes: int | None = None
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize identifiers and validate size metadata."""
        object.__setattr__(self, "id", _identifier(self.id, "id"))
        object.__setattr__(self, "filename", _identifier(self.filename, "filename"))
        object.__setattr__(self, "purpose", _identifier(self.purpose, "purpose"))
        if self.bytes is not None and self.bytes < 0:
            raise ValueError("bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class VectorStoreReference:
    """OpenAI vector-store identity and optional SDK resource.

    Parameters
    ----------
    id : str
        Vector-store identifier.
    name : str or None, default=None
        Human-readable store name when available.
    raw : object or None, default=None
        Underlying official SDK vector-store resource.
    """

    id: str
    name: str | None = None
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize the vector-store identity."""
        object.__setattr__(self, "id", _identifier(self.id, "id"))
        if self.name is not None:
            object.__setattr__(self, "name", _identifier(self.name, "name"))


@dataclass(frozen=True, slots=True)
class RetrievalOperationResult(Generic[ResourceT]):
    """Typed outcome that preserves the original SDK response.

    Parameters
    ----------
    operation : str
        Stable operation name such as ``files.upload``.
    resource : ResourceT or None
        Normalized resource produced or affected by the operation.
    succeeded : bool
        Whether the operation completed successfully.
    raw : object or None, default=None
        Original SDK response or deletion object.
    error : BaseException or None, default=None
        Original exception for non-raising batch or cleanup workflows.
    """

    operation: str
    resource: ResourceT | None
    succeeded: bool
    raw: object | None = field(default=None, repr=False, compare=False)
    error: BaseException | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate operation identity and outcome consistency."""
        object.__setattr__(self, "operation", _identifier(self.operation, "operation"))
        if self.succeeded and self.error is not None:
            raise ValueError("successful operations cannot contain an error")


@dataclass(frozen=True, slots=True)
class RetrievalSearchContent:
    """One text fragment returned from vector-store search."""

    text: str
    type: str = "text"

    def __post_init__(self) -> None:
        """Require non-empty content and content type."""
        object.__setattr__(self, "text", _identifier(self.text, "text"))
        object.__setattr__(self, "type", _identifier(self.type, "type"))


@dataclass(frozen=True, slots=True)
class RetrievalSearchResult:
    """Normalized vector-store search result with raw SDK access.

    Parameters
    ----------
    file_id : str
        Source file identifier.
    filename : str
        Source filename.
    score : float
        Relevance score reported by the API.
    content : tuple[RetrievalSearchContent, ...]
        Ordered returned content fragments.
    attributes : Mapping[str, AttributeValue], default={}
        Source attributes copied from the API result.
    raw : object or None, default=None
        Original SDK result item.
    """

    file_id: str
    filename: str
    score: float
    content: tuple[RetrievalSearchContent, ...]
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize identity, content, and copied attributes."""
        object.__setattr__(self, "file_id", _identifier(self.file_id, "file_id"))
        object.__setattr__(self, "filename", _identifier(self.filename, "filename"))
        if not self.content:
            raise ValueError("content must not be empty")
        object.__setattr__(self, "content", tuple(self.content))
        object.__setattr__(self, "attributes", dict(self.attributes))


@dataclass(frozen=True, slots=True)
class RetrievalSearchPage:
    """Normalized search page with pagination and raw SDK access."""

    query: tuple[str, ...]
    data: tuple[RetrievalSearchResult, ...]
    has_more: bool = False
    next_page: str | None = None
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize query values, result order, and cursor."""
        query = tuple(_identifier(value, "query") for value in self.query)
        if not query:
            raise ValueError("query must not be empty")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "data", tuple(self.data))
        if self.next_page is not None:
            object.__setattr__(
                self,
                "next_page",
                _identifier(self.next_page, "next_page"),
            )


@dataclass(frozen=True, slots=True)
class FileSearchConfig:
    """Explicit Responses and Agents File Search tool configuration.

    Parameters
    ----------
    vector_store_ids : tuple[str, ...]
        Existing vector stores to search.
    max_num_results : int or None, default=None
        Maximum results requested from File Search.
    filters : Mapping[str, Any] or None, default=None
        Official File Search attribute filter object.
    ranking_options : Mapping[str, Any] or None, default=None
        Official ranking options passed through unchanged.
    include_search_results : bool, default=False
        Whether a Responses request should include raw search results.
    """

    vector_store_ids: tuple[str, ...]
    max_num_results: int | None = None
    filters: Mapping[str, Any] | None = None
    ranking_options: Mapping[str, Any] | None = None
    include_search_results: bool = False

    def __post_init__(self) -> None:
        """Normalize store identifiers and copy mutable mappings."""
        identifiers = tuple(
            dict.fromkeys(_identifier(value, "vector_store_id") for value in self.vector_store_ids)
        )
        if not identifiers:
            raise ValueError("vector_store_ids must not be empty")
        if self.max_num_results is not None and self.max_num_results < 1:
            raise ValueError("max_num_results must be positive")
        object.__setattr__(self, "vector_store_ids", identifiers)
        if self.filters is not None:
            object.__setattr__(self, "filters", dict(self.filters))
        if self.ranking_options is not None:
            object.__setattr__(self, "ranking_options", dict(self.ranking_options))

    def as_tool(self) -> dict[str, Any]:
        """Return official SDK-shaped File Search tool configuration."""
        tool: dict[str, Any] = {
            "type": "file_search",
            "vector_store_ids": list(self.vector_store_ids),
        }
        if self.max_num_results is not None:
            tool["max_num_results"] = self.max_num_results
        if self.filters is not None:
            tool["filters"] = dict(self.filters)
        if self.ranking_options is not None:
            tool["ranking_options"] = dict(self.ranking_options)
        return tool

    def response_includes(self) -> tuple[str, ...]:
        """Return optional Responses include values for raw search results."""
        if self.include_search_results:
            return ("file_search_call.results",)
        return ()


@runtime_checkable
class SyncRetrievalClient(Protocol):
    """Dependency-injected synchronous retrieval lifecycle contract."""

    @property
    def sdk_client(self) -> object:
        """Return the underlying official OpenAI client."""
        ...

    def upload_file(
        self,
        source: FileSource,
        *,
        purpose: str,
        expires_after: Mapping[str, Any] | None = None,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Upload one file without implying vector-store attachment."""
        ...

    def create_vector_store(
        self,
        *,
        name: str,
        file_ids: Sequence[str] = (),
        attributes: Mapping[str, AttributeValue] | None = None,
        expires_after: Mapping[str, Any] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Create one vector store and preserve the SDK resource."""
        ...

    def attach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Attach an existing file to an existing vector store."""
        ...

    def detach_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Remove a file from a store without deleting the Files resource."""
        ...

    def search(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        *,
        filters: Mapping[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: Mapping[str, Any] | None = None,
        rewrite_query: bool | None = None,
    ) -> RetrievalSearchPage:
        """Search one vector store and return normalized results."""
        ...

    def delete_file(self, file_id: str) -> RetrievalOperationResult[UploadedFile]:
        """Delete the underlying Files resource explicitly."""
        ...

    def delete_vector_store(
        self,
        vector_store_id: str,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Delete a vector store explicitly."""
        ...


@runtime_checkable
class AsyncRetrievalClient(Protocol):
    """Dependency-injected asynchronous retrieval lifecycle contract."""

    @property
    def sdk_client(self) -> object:
        """Return the underlying official asynchronous OpenAI client."""
        ...

    def upload_file(
        self,
        source: FileSource,
        *,
        purpose: str,
        expires_after: Mapping[str, Any] | None = None,
    ) -> Awaitable[RetrievalOperationResult[UploadedFile]]:
        """Upload one file without implying vector-store attachment."""
        ...

    def create_vector_store(
        self,
        *,
        name: str,
        file_ids: Sequence[str] = (),
        attributes: Mapping[str, AttributeValue] | None = None,
        expires_after: Mapping[str, Any] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
    ) -> Awaitable[RetrievalOperationResult[VectorStoreReference]]:
        """Create one vector store and preserve the SDK resource."""
        ...

    def attach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
    ) -> Awaitable[RetrievalOperationResult[UploadedFile]]:
        """Attach an existing file to an existing vector store."""
        ...

    def detach_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> Awaitable[RetrievalOperationResult[UploadedFile]]:
        """Remove a file from a store without deleting the Files resource."""
        ...

    def search(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        *,
        filters: Mapping[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: Mapping[str, Any] | None = None,
        rewrite_query: bool | None = None,
    ) -> Awaitable[RetrievalSearchPage]:
        """Search one vector store and return normalized results."""
        ...

    def delete_file(
        self,
        file_id: str,
    ) -> Awaitable[RetrievalOperationResult[UploadedFile]]:
        """Delete the underlying Files resource explicitly."""
        ...

    def delete_vector_store(
        self,
        vector_store_id: str,
    ) -> Awaitable[RetrievalOperationResult[VectorStoreReference]]:
        """Delete a vector store explicitly."""
        ...


__all__ = [
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
]

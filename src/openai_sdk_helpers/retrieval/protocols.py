"""Runtime-checkable retrieval lifecycle protocols."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from openai_sdk_helpers.runtime import OperationContext

from .contracts import (
    AttributeValue,
    FileSource,
    RetrievalOperationResult,
    UploadedFile,
    VectorStoreReference,
)
from .lifecycle import (
    PollingConfig,
    RetrievalBatchResult,
    VectorStoreFileReference,
)


@runtime_checkable
class SyncRetrievalLifecycleClient(Protocol):
    """Synchronous Files and Vector Stores lifecycle contract.

    Methods
    -------
    upload_file(source, *, purpose, expires_after=None, operation_context=None)
        Upload one file without implying vector-store attachment.
    upload_files(sources, *, purpose, expires_after=None, continue_on_error=True, operation_context=None)
        Upload files sequentially while preserving ordered outcomes.
    create_vector_store(*, name, file_ids=(), metadata=None, expires_after=None, chunking_strategy=None, description=None, operation_context=None)
        Create one vector store.
    retrieve_vector_store(vector_store_id, *, operation_context=None)
        Retrieve one vector store.
    list_vector_stores(*, limit=None, order=None, after=None, before=None, operation_context=None)
        List one page of vector stores.
    update_vector_store(vector_store_id, *, name=None, metadata=None, expires_after=None, operation_context=None)
        Update explicit vector-store metadata.
    attach_file(vector_store_id, file_id, *, attributes=None, chunking_strategy=None, polling=None, operation_context=None)
        Attach and poll an existing file.
    upload_and_poll(vector_store_id, source, *, attributes=None, chunking_strategy=None, polling=None, operation_context=None)
        Upload through the vector-store helper and poll.
    detach_file(vector_store_id, file_id, *, operation_context=None)
        Detach a file without deleting the Files resource.
    delete_file(file_id, *, operation_context=None)
        Delete an underlying Files resource explicitly.
    delete_vector_store(vector_store_id, *, operation_context=None)
        Delete a vector store explicitly.
    """

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
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Upload one file without implying vector-store attachment."""
        ...

    def upload_files(
        self,
        sources: Sequence[FileSource],
        *,
        purpose: str,
        expires_after: Mapping[str, Any] | None = None,
        continue_on_error: bool = True,
        operation_context: OperationContext | None = None,
    ) -> RetrievalBatchResult[UploadedFile]:
        """Upload files sequentially and preserve ordered outcomes."""
        ...

    def create_vector_store(
        self,
        *,
        name: str,
        file_ids: Sequence[str] = (),
        metadata: Mapping[str, str] | None = None,
        expires_after: Mapping[str, Any] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
        description: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Create one vector store."""
        ...

    def retrieve_vector_store(
        self,
        vector_store_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> VectorStoreReference:
        """Retrieve one vector store."""
        ...

    def list_vector_stores(
        self,
        *,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
        before: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> tuple[VectorStoreReference, ...]:
        """List one page of vector stores."""
        ...

    def update_vector_store(
        self,
        vector_store_id: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, str] | None = None,
        expires_after: Mapping[str, Any] | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Update explicit vector-store metadata."""
        ...

    def attach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
        polling: PollingConfig | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreFileReference]:
        """Attach and poll an existing file."""
        ...

    def upload_and_poll(
        self,
        vector_store_id: str,
        source: FileSource,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
        polling: PollingConfig | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreFileReference]:
        """Upload through the vector-store helper and poll."""
        ...

    def detach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreFileReference]:
        """Detach a file without deleting the Files resource."""
        ...

    def delete_file(
        self,
        file_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Delete an underlying Files resource explicitly."""
        ...

    def delete_vector_store(
        self,
        vector_store_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Delete a vector store explicitly."""
        ...


@runtime_checkable
class AsyncRetrievalLifecycleClient(Protocol):
    """Asynchronous Files and Vector Stores lifecycle contract.

    Methods
    -------
    upload_file(source, *, purpose, expires_after=None, operation_context=None)
        Upload one file without implying vector-store attachment.
    upload_files(sources, *, purpose, expires_after=None, continue_on_error=True, operation_context=None)
        Upload files sequentially while preserving ordered outcomes.
    create_vector_store(*, name, file_ids=(), metadata=None, expires_after=None, chunking_strategy=None, description=None, operation_context=None)
        Create one vector store.
    retrieve_vector_store(vector_store_id, *, operation_context=None)
        Retrieve one vector store.
    list_vector_stores(*, limit=None, order=None, after=None, before=None, operation_context=None)
        List one page of vector stores.
    update_vector_store(vector_store_id, *, name=None, metadata=None, expires_after=None, operation_context=None)
        Update explicit vector-store metadata.
    attach_file(vector_store_id, file_id, *, attributes=None, chunking_strategy=None, polling=None, operation_context=None)
        Attach and poll an existing file.
    upload_and_poll(vector_store_id, source, *, attributes=None, chunking_strategy=None, polling=None, operation_context=None)
        Upload through the vector-store helper and poll.
    detach_file(vector_store_id, file_id, *, operation_context=None)
        Detach a file without deleting the Files resource.
    delete_file(file_id, *, operation_context=None)
        Delete an underlying Files resource explicitly.
    delete_vector_store(vector_store_id, *, operation_context=None)
        Delete a vector store explicitly.
    """

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
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalOperationResult[UploadedFile]]:
        """Upload one file without implying vector-store attachment."""
        ...

    def upload_files(
        self,
        sources: Sequence[FileSource],
        *,
        purpose: str,
        expires_after: Mapping[str, Any] | None = None,
        continue_on_error: bool = True,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalBatchResult[UploadedFile]]:
        """Upload files sequentially and preserve ordered outcomes."""
        ...

    def create_vector_store(
        self,
        *,
        name: str,
        file_ids: Sequence[str] = (),
        metadata: Mapping[str, str] | None = None,
        expires_after: Mapping[str, Any] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
        description: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalOperationResult[VectorStoreReference]]:
        """Create one vector store."""
        ...

    def retrieve_vector_store(
        self,
        vector_store_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[VectorStoreReference]:
        """Retrieve one vector store."""
        ...

    def list_vector_stores(
        self,
        *,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
        before: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[tuple[VectorStoreReference, ...]]:
        """List one page of vector stores."""
        ...

    def update_vector_store(
        self,
        vector_store_id: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, str] | None = None,
        expires_after: Mapping[str, Any] | None = None,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalOperationResult[VectorStoreReference]]:
        """Update explicit vector-store metadata."""
        ...

    def attach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
        polling: PollingConfig | None = None,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalOperationResult[VectorStoreFileReference]]:
        """Attach and poll an existing file."""
        ...

    def upload_and_poll(
        self,
        vector_store_id: str,
        source: FileSource,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
        polling: PollingConfig | None = None,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalOperationResult[VectorStoreFileReference]]:
        """Upload through the vector-store helper and poll."""
        ...

    def detach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalOperationResult[VectorStoreFileReference]]:
        """Detach a file without deleting the Files resource."""
        ...

    def delete_file(
        self,
        file_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalOperationResult[UploadedFile]]:
        """Delete an underlying Files resource explicitly."""
        ...

    def delete_vector_store(
        self,
        vector_store_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> Awaitable[RetrievalOperationResult[VectorStoreReference]]:
        """Delete a vector store explicitly."""
        ...


__all__ = ["AsyncRetrievalLifecycleClient", "SyncRetrievalLifecycleClient"]

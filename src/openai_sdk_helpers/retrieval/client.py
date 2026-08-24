"""Dependency-injected OpenAI retrieval lifecycle clients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openai_sdk_helpers.runtime import (
    OperationContext,
    run_observed_async,
    run_observed_sync,
)

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


def _required_identifier(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _source_filename(source: FileSource) -> str | None:
    if isinstance(source, (str, Path)):
        return Path(source).name
    name = getattr(source, "name", None)
    if isinstance(name, str):
        return Path(name).name
    return None


def _uploaded_file(raw: object, source: FileSource, purpose: str) -> UploadedFile:
    file_id = _required_identifier(str(getattr(raw, "id", "")), "file.id")
    filename = getattr(raw, "filename", None) or _source_filename(source) or file_id
    raw_purpose = getattr(raw, "purpose", None) or purpose
    byte_count = getattr(raw, "bytes", None)
    return UploadedFile(
        id=file_id,
        filename=str(filename),
        purpose=str(raw_purpose),
        bytes=byte_count if isinstance(byte_count, int) else None,
        raw=raw,
    )


def _vector_store(raw: object, fallback_id: str | None = None) -> VectorStoreReference:
    store_id = getattr(raw, "id", None) or fallback_id or ""
    name = getattr(raw, "name", None)
    return VectorStoreReference(
        id=_required_identifier(str(store_id), "vector_store.id"),
        name=str(name) if name else None,
        raw=raw,
    )


def _vector_store_file(
    raw: object,
    *,
    vector_store_id: str,
    fallback_file_id: str | None = None,
) -> VectorStoreFileReference:
    file_id = getattr(raw, "id", None) or fallback_file_id or ""
    status = getattr(raw, "status", None)
    return VectorStoreFileReference(
        file_id=_required_identifier(str(file_id), "vector_store_file.id"),
        vector_store_id=vector_store_id,
        status=str(status) if status else None,
        last_error=getattr(raw, "last_error", None),
        raw=raw,
    )


def _optional_kwargs(**values: object | None) -> dict[str, object]:
    return {name: value for name, value in values.items() if value is not None}


class OpenAIRetrievalClient:
    """Synchronous lifecycle wrapper around an injected official OpenAI client.

    Parameters
    ----------
    client : object
        Configured official ``OpenAI`` client. The wrapper does not close it.

    Methods
    -------
    upload_file(source, *, purpose, expires_after=None, operation_context=None)
        Upload one file without closing caller-owned handles.
    upload_files(sources, *, purpose, expires_after=None, continue_on_error=True, operation_context=None)
        Upload files sequentially and preserve ordered outcomes.
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

    def __init__(self, client: object) -> None:
        if not hasattr(client, "files") or not hasattr(client, "vector_stores"):
            raise TypeError("client must expose files and vector_stores resources")
        self._client = client

    @property
    def sdk_client(self) -> object:
        """Return the injected official OpenAI client."""
        return self._client

    def upload_file(
        self,
        source: FileSource,
        *,
        purpose: str,
        expires_after: Mapping[str, Any] | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Upload one file without closing caller-owned file handles."""
        normalized_purpose = _required_identifier(purpose, "purpose")

        def execute() -> RetrievalOperationResult[UploadedFile]:
            kwargs = _optional_kwargs(
                file=source,
                purpose=normalized_purpose,
                expires_after=(
                    dict(expires_after) if expires_after is not None else None
                ),
            )
            raw = self._client.files.create(**kwargs)  # type: ignore[attr-defined]
            resource = _uploaded_file(raw, source, normalized_purpose)
            return RetrievalOperationResult(
                operation="files.upload",
                resource=resource,
                succeeded=True,
                raw=raw,
            )

        return run_observed_sync(operation_context, execute)

    def upload_files(
        self,
        sources: Sequence[FileSource],
        *,
        purpose: str,
        expires_after: Mapping[str, Any] | None = None,
        continue_on_error: bool = True,
        operation_context: OperationContext | None = None,
    ) -> RetrievalBatchResult[UploadedFile]:
        """Upload files sequentially and preserve input order and failures."""

        def execute() -> RetrievalBatchResult[UploadedFile]:
            results: list[RetrievalOperationResult[UploadedFile]] = []
            for source in sources:
                try:
                    results.append(
                        self.upload_file(
                            source,
                            purpose=purpose,
                            expires_after=expires_after,
                        )
                    )
                except Exception as error:
                    if not continue_on_error:
                        raise
                    results.append(
                        RetrievalOperationResult(
                            operation="files.upload",
                            resource=None,
                            succeeded=False,
                            error=error,
                        )
                    )
            return RetrievalBatchResult(tuple(results))

        return run_observed_sync(operation_context, execute)

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
        """Create one vector store without assuming cleanup ownership."""
        normalized_name = _required_identifier(name, "name")
        normalized_file_ids = tuple(
            _required_identifier(file_id, "file_id") for file_id in file_ids
        )

        def execute() -> RetrievalOperationResult[VectorStoreReference]:
            kwargs = _optional_kwargs(
                name=normalized_name,
                file_ids=normalized_file_ids or None,
                metadata=dict(metadata) if metadata is not None else None,
                expires_after=(
                    dict(expires_after) if expires_after is not None else None
                ),
                chunking_strategy=(
                    dict(chunking_strategy) if chunking_strategy is not None else None
                ),
                description=description,
            )
            raw = self._client.vector_stores.create(**kwargs)  # type: ignore[attr-defined]
            resource = _vector_store(raw)
            return RetrievalOperationResult(
                operation="vector_stores.create",
                resource=resource,
                succeeded=True,
                raw=raw,
            )

        return run_observed_sync(operation_context, execute)

    def retrieve_vector_store(
        self,
        vector_store_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> VectorStoreReference:
        """Retrieve one vector store and preserve the SDK resource."""
        normalized_id = _required_identifier(vector_store_id, "vector_store_id")
        return run_observed_sync(
            operation_context,
            lambda: _vector_store(
                self._client.vector_stores.retrieve(normalized_id),  # type: ignore[attr-defined]
                normalized_id,
            ),
        )

    def list_vector_stores(
        self,
        *,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
        before: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> tuple[VectorStoreReference, ...]:
        """List one page of vector stores in SDK order."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")

        def execute() -> tuple[VectorStoreReference, ...]:
            kwargs = _optional_kwargs(
                limit=limit,
                order=order,
                after=after,
                before=before,
            )
            raw_page = self._client.vector_stores.list(**kwargs)  # type: ignore[attr-defined]
            return tuple(_vector_store(raw) for raw in getattr(raw_page, "data", ()))

        return run_observed_sync(operation_context, execute)

    def update_vector_store(
        self,
        vector_store_id: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, str] | None = None,
        expires_after: Mapping[str, Any] | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Update explicit mutable vector-store metadata."""
        normalized_id = _required_identifier(vector_store_id, "vector_store_id")
        if name is not None:
            name = _required_identifier(name, "name")

        def execute() -> RetrievalOperationResult[VectorStoreReference]:
            kwargs = _optional_kwargs(
                name=name,
                metadata=dict(metadata) if metadata is not None else None,
                expires_after=(
                    dict(expires_after) if expires_after is not None else None
                ),
            )
            raw = self._client.vector_stores.update(  # type: ignore[attr-defined]
                normalized_id,
                **kwargs,
            )
            return RetrievalOperationResult(
                operation="vector_stores.update",
                resource=_vector_store(raw, normalized_id),
                succeeded=True,
                raw=raw,
            )

        return run_observed_sync(operation_context, execute)

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
        """Attach an existing file and poll until the SDK returns a terminal state."""
        normalized_store_id = _required_identifier(
            vector_store_id,
            "vector_store_id",
        )
        normalized_file_id = _required_identifier(file_id, "file_id")
        polling = polling or PollingConfig()

        def execute() -> RetrievalOperationResult[VectorStoreFileReference]:
            kwargs = _optional_kwargs(
                vector_store_id=normalized_store_id,
                file_id=normalized_file_id,
                attributes=dict(attributes) if attributes is not None else None,
                chunking_strategy=(
                    dict(chunking_strategy) if chunking_strategy is not None else None
                ),
            )
            kwargs.update(polling.sdk_kwargs())
            raw = self._client.vector_stores.files.create_and_poll(  # type: ignore[attr-defined]
                **kwargs
            )
            resource = _vector_store_file(
                raw,
                vector_store_id=normalized_store_id,
                fallback_file_id=normalized_file_id,
            )
            polling.validate_terminal_status(resource.status)
            return RetrievalOperationResult(
                operation="vector_stores.files.attach",
                resource=resource,
                succeeded=resource.status == "completed",
                raw=raw,
            )

        return run_observed_sync(operation_context, execute)

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
        """Upload one file through the vector-store helper and poll explicitly."""
        normalized_store_id = _required_identifier(
            vector_store_id,
            "vector_store_id",
        )
        polling = polling or PollingConfig()

        def execute() -> RetrievalOperationResult[VectorStoreFileReference]:
            kwargs = _optional_kwargs(
                vector_store_id=normalized_store_id,
                file=source,
                attributes=dict(attributes) if attributes is not None else None,
                chunking_strategy=(
                    dict(chunking_strategy) if chunking_strategy is not None else None
                ),
            )
            kwargs.update(polling.sdk_kwargs())
            raw = self._client.vector_stores.files.upload_and_poll(  # type: ignore[attr-defined]
                **kwargs
            )
            resource = _vector_store_file(
                raw,
                vector_store_id=normalized_store_id,
            )
            polling.validate_terminal_status(resource.status)
            return RetrievalOperationResult(
                operation="vector_stores.files.upload_and_poll",
                resource=resource,
                succeeded=resource.status == "completed",
                raw=raw,
            )

        return run_observed_sync(operation_context, execute)

    def detach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreFileReference]:
        """Detach a file without deleting the underlying Files resource."""
        normalized_store_id = _required_identifier(
            vector_store_id,
            "vector_store_id",
        )
        normalized_file_id = _required_identifier(file_id, "file_id")

        def execute() -> RetrievalOperationResult[VectorStoreFileReference]:
            raw = self._client.vector_stores.files.delete(  # type: ignore[attr-defined]
                normalized_file_id,
                vector_store_id=normalized_store_id,
            )
            return RetrievalOperationResult(
                operation="vector_stores.files.detach",
                resource=VectorStoreFileReference(
                    file_id=normalized_file_id,
                    vector_store_id=normalized_store_id,
                    raw=raw,
                ),
                succeeded=bool(getattr(raw, "deleted", True)),
                raw=raw,
            )

        return run_observed_sync(operation_context, execute)

    def delete_file(
        self,
        file_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Delete the underlying Files resource only when explicitly requested."""
        normalized_id = _required_identifier(file_id, "file_id")

        def execute() -> RetrievalOperationResult[UploadedFile]:
            raw = self._client.files.delete(normalized_id)  # type: ignore[attr-defined]
            return RetrievalOperationResult(
                operation="files.delete",
                resource=None,
                succeeded=bool(getattr(raw, "deleted", True)),
                raw=raw,
            )

        return run_observed_sync(operation_context, execute)

    def delete_vector_store(
        self,
        vector_store_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Delete one vector store only when explicitly requested."""
        normalized_id = _required_identifier(vector_store_id, "vector_store_id")

        def execute() -> RetrievalOperationResult[VectorStoreReference]:
            raw = self._client.vector_stores.delete(normalized_id)  # type: ignore[attr-defined]
            return RetrievalOperationResult(
                operation="vector_stores.delete",
                resource=VectorStoreReference(id=normalized_id, raw=raw),
                succeeded=bool(getattr(raw, "deleted", True)),
                raw=raw,
            )

        return run_observed_sync(operation_context, execute)


class AsyncOpenAIRetrievalClient:
    """Asynchronous lifecycle wrapper around an injected ``AsyncOpenAI`` client.

    Parameters
    ----------
    client : object
        Configured official ``AsyncOpenAI`` client. The wrapper does not close it.

    Methods
    -------
    upload_file(source, *, purpose, expires_after=None, operation_context=None)
        Upload one file without closing caller-owned handles.
    upload_files(sources, *, purpose, expires_after=None, continue_on_error=True, operation_context=None)
        Upload files sequentially and preserve ordered outcomes.
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

    def __init__(self, client: object) -> None:
        if not hasattr(client, "files") or not hasattr(client, "vector_stores"):
            raise TypeError("client must expose files and vector_stores resources")
        self._client = client

    @property
    def sdk_client(self) -> object:
        """Return the injected official asynchronous OpenAI client."""
        return self._client

    async def upload_file(
        self,
        source: FileSource,
        *,
        purpose: str,
        expires_after: Mapping[str, Any] | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Upload one file without closing caller-owned file handles."""
        normalized_purpose = _required_identifier(purpose, "purpose")

        async def execute() -> RetrievalOperationResult[UploadedFile]:
            kwargs = _optional_kwargs(
                file=source,
                purpose=normalized_purpose,
                expires_after=(
                    dict(expires_after) if expires_after is not None else None
                ),
            )
            raw = await self._client.files.create(**kwargs)  # type: ignore[attr-defined]
            resource = _uploaded_file(raw, source, normalized_purpose)
            return RetrievalOperationResult(
                operation="files.upload",
                resource=resource,
                succeeded=True,
                raw=raw,
            )

        return await run_observed_async(operation_context, execute)

    async def upload_files(
        self,
        sources: Sequence[FileSource],
        *,
        purpose: str,
        expires_after: Mapping[str, Any] | None = None,
        continue_on_error: bool = True,
        operation_context: OperationContext | None = None,
    ) -> RetrievalBatchResult[UploadedFile]:
        """Upload files sequentially and preserve input order and failures."""

        async def execute() -> RetrievalBatchResult[UploadedFile]:
            results: list[RetrievalOperationResult[UploadedFile]] = []
            for source in sources:
                try:
                    results.append(
                        await self.upload_file(
                            source,
                            purpose=purpose,
                            expires_after=expires_after,
                        )
                    )
                except Exception as error:
                    if not continue_on_error:
                        raise
                    results.append(
                        RetrievalOperationResult(
                            operation="files.upload",
                            resource=None,
                            succeeded=False,
                            error=error,
                        )
                    )
            return RetrievalBatchResult(tuple(results))

        return await run_observed_async(operation_context, execute)

    async def create_vector_store(
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
        """Create one vector store without assuming cleanup ownership."""
        normalized_name = _required_identifier(name, "name")
        normalized_file_ids = tuple(
            _required_identifier(file_id, "file_id") for file_id in file_ids
        )

        async def execute() -> RetrievalOperationResult[VectorStoreReference]:
            kwargs = _optional_kwargs(
                name=normalized_name,
                file_ids=normalized_file_ids or None,
                metadata=dict(metadata) if metadata is not None else None,
                expires_after=(
                    dict(expires_after) if expires_after is not None else None
                ),
                chunking_strategy=(
                    dict(chunking_strategy) if chunking_strategy is not None else None
                ),
                description=description,
            )
            raw = await self._client.vector_stores.create(**kwargs)  # type: ignore[attr-defined]
            return RetrievalOperationResult(
                operation="vector_stores.create",
                resource=_vector_store(raw),
                succeeded=True,
                raw=raw,
            )

        return await run_observed_async(operation_context, execute)

    async def retrieve_vector_store(
        self,
        vector_store_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> VectorStoreReference:
        """Retrieve one vector store and preserve the SDK resource."""
        normalized_id = _required_identifier(vector_store_id, "vector_store_id")

        async def execute() -> VectorStoreReference:
            raw = await self._client.vector_stores.retrieve(  # type: ignore[attr-defined]
                normalized_id
            )
            return _vector_store(raw, normalized_id)

        return await run_observed_async(operation_context, execute)

    async def list_vector_stores(
        self,
        *,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
        before: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> tuple[VectorStoreReference, ...]:
        """List one page of vector stores in SDK order."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")

        async def execute() -> tuple[VectorStoreReference, ...]:
            kwargs = _optional_kwargs(
                limit=limit,
                order=order,
                after=after,
                before=before,
            )
            raw_page = await self._client.vector_stores.list(**kwargs)  # type: ignore[attr-defined]
            return tuple(_vector_store(raw) for raw in getattr(raw_page, "data", ()))

        return await run_observed_async(operation_context, execute)

    async def update_vector_store(
        self,
        vector_store_id: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, str] | None = None,
        expires_after: Mapping[str, Any] | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Update explicit mutable vector-store metadata."""
        normalized_id = _required_identifier(vector_store_id, "vector_store_id")
        if name is not None:
            name = _required_identifier(name, "name")

        async def execute() -> RetrievalOperationResult[VectorStoreReference]:
            kwargs = _optional_kwargs(
                name=name,
                metadata=dict(metadata) if metadata is not None else None,
                expires_after=(
                    dict(expires_after) if expires_after is not None else None
                ),
            )
            raw = await self._client.vector_stores.update(  # type: ignore[attr-defined]
                normalized_id,
                **kwargs,
            )
            return RetrievalOperationResult(
                operation="vector_stores.update",
                resource=_vector_store(raw, normalized_id),
                succeeded=True,
                raw=raw,
            )

        return await run_observed_async(operation_context, execute)

    async def attach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
        polling: PollingConfig | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreFileReference]:
        """Attach an existing file and poll until the SDK returns a terminal state."""
        normalized_store_id = _required_identifier(
            vector_store_id,
            "vector_store_id",
        )
        normalized_file_id = _required_identifier(file_id, "file_id")
        polling = polling or PollingConfig()

        async def execute() -> RetrievalOperationResult[VectorStoreFileReference]:
            kwargs = _optional_kwargs(
                vector_store_id=normalized_store_id,
                file_id=normalized_file_id,
                attributes=dict(attributes) if attributes is not None else None,
                chunking_strategy=(
                    dict(chunking_strategy) if chunking_strategy is not None else None
                ),
            )
            kwargs.update(polling.sdk_kwargs())
            raw = await self._client.vector_stores.files.create_and_poll(  # type: ignore[attr-defined]
                **kwargs
            )
            resource = _vector_store_file(
                raw,
                vector_store_id=normalized_store_id,
                fallback_file_id=normalized_file_id,
            )
            polling.validate_terminal_status(resource.status)
            return RetrievalOperationResult(
                operation="vector_stores.files.attach",
                resource=resource,
                succeeded=resource.status == "completed",
                raw=raw,
            )

        return await run_observed_async(operation_context, execute)

    async def upload_and_poll(
        self,
        vector_store_id: str,
        source: FileSource,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        chunking_strategy: Mapping[str, Any] | None = None,
        polling: PollingConfig | None = None,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreFileReference]:
        """Upload one file through the vector-store helper and poll explicitly."""
        normalized_store_id = _required_identifier(
            vector_store_id,
            "vector_store_id",
        )
        polling = polling or PollingConfig()

        async def execute() -> RetrievalOperationResult[VectorStoreFileReference]:
            kwargs = _optional_kwargs(
                vector_store_id=normalized_store_id,
                file=source,
                attributes=dict(attributes) if attributes is not None else None,
                chunking_strategy=(
                    dict(chunking_strategy) if chunking_strategy is not None else None
                ),
            )
            kwargs.update(polling.sdk_kwargs())
            raw = await self._client.vector_stores.files.upload_and_poll(  # type: ignore[attr-defined]
                **kwargs
            )
            resource = _vector_store_file(
                raw,
                vector_store_id=normalized_store_id,
            )
            polling.validate_terminal_status(resource.status)
            return RetrievalOperationResult(
                operation="vector_stores.files.upload_and_poll",
                resource=resource,
                succeeded=resource.status == "completed",
                raw=raw,
            )

        return await run_observed_async(operation_context, execute)

    async def detach_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreFileReference]:
        """Detach a file without deleting the underlying Files resource."""
        normalized_store_id = _required_identifier(
            vector_store_id,
            "vector_store_id",
        )
        normalized_file_id = _required_identifier(file_id, "file_id")

        async def execute() -> RetrievalOperationResult[VectorStoreFileReference]:
            raw = await self._client.vector_stores.files.delete(  # type: ignore[attr-defined]
                normalized_file_id,
                vector_store_id=normalized_store_id,
            )
            return RetrievalOperationResult(
                operation="vector_stores.files.detach",
                resource=VectorStoreFileReference(
                    file_id=normalized_file_id,
                    vector_store_id=normalized_store_id,
                    raw=raw,
                ),
                succeeded=bool(getattr(raw, "deleted", True)),
                raw=raw,
            )

        return await run_observed_async(operation_context, execute)

    async def delete_file(
        self,
        file_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[UploadedFile]:
        """Delete the underlying Files resource only when explicitly requested."""
        normalized_id = _required_identifier(file_id, "file_id")

        async def execute() -> RetrievalOperationResult[UploadedFile]:
            raw = await self._client.files.delete(normalized_id)  # type: ignore[attr-defined]
            return RetrievalOperationResult(
                operation="files.delete",
                resource=None,
                succeeded=bool(getattr(raw, "deleted", True)),
                raw=raw,
            )

        return await run_observed_async(operation_context, execute)

    async def delete_vector_store(
        self,
        vector_store_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> RetrievalOperationResult[VectorStoreReference]:
        """Delete one vector store only when explicitly requested."""
        normalized_id = _required_identifier(vector_store_id, "vector_store_id")

        async def execute() -> RetrievalOperationResult[VectorStoreReference]:
            raw = await self._client.vector_stores.delete(  # type: ignore[attr-defined]
                normalized_id
            )
            return RetrievalOperationResult(
                operation="vector_stores.delete",
                resource=VectorStoreReference(id=normalized_id, raw=raw),
                succeeded=bool(getattr(raw, "deleted", True)),
                raw=raw,
            )

        return await run_observed_async(operation_context, execute)


__all__ = ["AsyncOpenAIRetrievalClient", "OpenAIRetrievalClient"]

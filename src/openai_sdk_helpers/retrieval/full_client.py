"""Full retrieval clients combining lifecycle and direct search."""

from __future__ import annotations

from .client import (
    AsyncOpenAIRetrievalClient as AsyncRetrievalLifecycleClient,
)
from .client import OpenAIRetrievalClient as RetrievalLifecycleClient
from .search import AsyncSearchMixin, SyncSearchMixin


class OpenAIRetrievalClient(SyncSearchMixin, RetrievalLifecycleClient):
    """Synchronous OpenAI retrieval lifecycle and direct search client.

    Methods
    -------
    search(vector_store_id, query, **options)
        Search one vector store and normalize results in SDK order.
    upload_file(source, *, purpose, **options)
        Upload one file without closing caller-owned handles.
    upload_files(sources, *, purpose, **options)
        Upload files sequentially and preserve ordered outcomes.
    create_vector_store(**options)
        Create one vector store.
    retrieve_vector_store(vector_store_id, **options)
        Retrieve one vector store.
    list_vector_stores(**options)
        List one page of vector stores.
    update_vector_store(vector_store_id, **options)
        Update explicit vector-store metadata.
    attach_file(vector_store_id, file_id, **options)
        Attach and poll an existing file.
    upload_and_poll(vector_store_id, source, **options)
        Upload through the vector-store helper and poll.
    detach_file(vector_store_id, file_id, **options)
        Detach a file without deleting the Files resource.
    delete_file(file_id, **options)
        Delete an underlying Files resource explicitly.
    delete_vector_store(vector_store_id, **options)
        Delete a vector store explicitly.
    """


class AsyncOpenAIRetrievalClient(
    AsyncSearchMixin,
    AsyncRetrievalLifecycleClient,
):
    """Asynchronous OpenAI retrieval lifecycle and direct search client.

    Methods
    -------
    search(vector_store_id, query, **options)
        Search one vector store and normalize results in SDK order.
    upload_file(source, *, purpose, **options)
        Upload one file without closing caller-owned handles.
    upload_files(sources, *, purpose, **options)
        Upload files sequentially and preserve ordered outcomes.
    create_vector_store(**options)
        Create one vector store.
    retrieve_vector_store(vector_store_id, **options)
        Retrieve one vector store.
    list_vector_stores(**options)
        List one page of vector stores.
    update_vector_store(vector_store_id, **options)
        Update explicit vector-store metadata.
    attach_file(vector_store_id, file_id, **options)
        Attach and poll an existing file.
    upload_and_poll(vector_store_id, source, **options)
        Upload through the vector-store helper and poll.
    detach_file(vector_store_id, file_id, **options)
        Detach a file without deleting the Files resource.
    delete_file(file_id, **options)
        Delete an underlying Files resource explicitly.
    delete_vector_store(vector_store_id, **options)
        Delete a vector store explicitly.
    """


__all__ = ["AsyncOpenAIRetrievalClient", "OpenAIRetrievalClient"]

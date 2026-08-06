"""Network-free tests for retrieval lifecycle clients."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openai_sdk_helpers.retrieval import (
    AsyncOpenAIRetrievalClient,
    OpenAIRetrievalClient,
    PollingConfig,
)


class NamedBuffer(io.BytesIO):
    """In-memory caller-owned file with a filename."""

    name = "buffer.txt"


class SyncFiles:
    """SDK-shaped synchronous Files resource."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def create(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        if kwargs["file"] == "bad.txt":
            raise RuntimeError("upload failed")
        filename = Path(getattr(kwargs["file"], "name", kwargs["file"])).name
        return SimpleNamespace(
            id=f"file_{len(self.created)}",
            filename=filename,
            purpose=kwargs["purpose"],
            bytes=12,
        )

    def delete(self, file_id: str) -> object:
        self.deleted.append(file_id)
        return SimpleNamespace(id=file_id, deleted=True)


class SyncVectorStoreFiles:
    """SDK-shaped synchronous vector-store files resource."""

    def __init__(self) -> None:
        self.attach_calls: list[dict[str, Any]] = []
        self.upload_calls: list[dict[str, Any]] = []
        self.detach_calls: list[tuple[str, str]] = []
        self.status = "completed"
        self.error: BaseException | None = None

    def create_and_poll(self, **kwargs: Any) -> object:
        self.attach_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            id=kwargs["file_id"],
            status=self.status,
            last_error=None,
        )

    def upload_and_poll(self, **kwargs: Any) -> object:
        self.upload_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(id="file_uploaded", status=self.status, last_error=None)

    def delete(self, file_id: str, *, vector_store_id: str) -> object:
        self.detach_calls.append((vector_store_id, file_id))
        return SimpleNamespace(id=file_id, deleted=True)


class SyncVectorStores:
    """SDK-shaped synchronous Vector Stores resource."""

    def __init__(self) -> None:
        self.files = SyncVectorStoreFiles()
        self.created: list[dict[str, Any]] = []
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []

    def create(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        return SimpleNamespace(id="vs_created", name=kwargs["name"])

    def retrieve(self, vector_store_id: str) -> object:
        return SimpleNamespace(id=vector_store_id, name="retrieved")

    def list(self, **_: Any) -> object:
        return SimpleNamespace(
            data=(
                SimpleNamespace(id="vs_1", name="one"),
                SimpleNamespace(id="vs_2", name="two"),
            )
        )

    def update(self, vector_store_id: str, **kwargs: Any) -> object:
        self.updated.append((vector_store_id, kwargs))
        return SimpleNamespace(id=vector_store_id, name=kwargs.get("name"))

    def delete(self, vector_store_id: str) -> object:
        self.deleted.append(vector_store_id)
        return SimpleNamespace(id=vector_store_id, deleted=True)


class SyncClient:
    """SDK-shaped synchronous OpenAI client."""

    def __init__(self) -> None:
        self.files = SyncFiles()
        self.vector_stores = SyncVectorStores()


class AsyncFiles:
    """SDK-shaped asynchronous Files resource."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    async def create(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        if kwargs["file"] == "bad.txt":
            raise RuntimeError("upload failed")
        return SimpleNamespace(
            id=f"file_{len(self.created)}",
            filename=Path(str(kwargs["file"])).name,
            purpose=kwargs["purpose"],
            bytes=3,
        )

    async def delete(self, file_id: str) -> object:
        self.deleted.append(file_id)
        return SimpleNamespace(id=file_id, deleted=True)


class AsyncVectorStoreFiles:
    """SDK-shaped asynchronous vector-store files resource."""

    async def create_and_poll(self, **kwargs: Any) -> object:
        return SimpleNamespace(
            id=kwargs["file_id"],
            status="completed",
            last_error=None,
        )

    async def upload_and_poll(self, **_: Any) -> object:
        return SimpleNamespace(id="file_uploaded", status="completed", last_error=None)

    async def delete(self, file_id: str, *, vector_store_id: str) -> object:
        return SimpleNamespace(id=file_id, vector_store_id=vector_store_id, deleted=True)


class AsyncVectorStores:
    """SDK-shaped asynchronous Vector Stores resource."""

    def __init__(self) -> None:
        self.files = AsyncVectorStoreFiles()

    async def create(self, **kwargs: Any) -> object:
        return SimpleNamespace(id="vs_created", name=kwargs["name"])

    async def retrieve(self, vector_store_id: str) -> object:
        return SimpleNamespace(id=vector_store_id, name="retrieved")

    async def list(self, **_: Any) -> object:
        return SimpleNamespace(data=(SimpleNamespace(id="vs_1", name="one"),))

    async def update(self, vector_store_id: str, **kwargs: Any) -> object:
        return SimpleNamespace(id=vector_store_id, name=kwargs.get("name"))

    async def delete(self, vector_store_id: str) -> object:
        return SimpleNamespace(id=vector_store_id, deleted=True)


class AsyncClient:
    """SDK-shaped asynchronous OpenAI client."""

    def __init__(self) -> None:
        self.files = AsyncFiles()
        self.vector_stores = AsyncVectorStores()


def test_upload_preserves_caller_file_handle() -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)
    source = NamedBuffer(b"hello")

    result = client.upload_file(source, purpose="user_data")

    assert result.succeeded is True
    assert result.resource is not None
    assert result.resource.filename == "buffer.txt"
    assert result.resource.raw is result.raw
    assert source.closed is False


def test_batch_upload_preserves_order_and_partial_failures() -> None:
    client = OpenAIRetrievalClient(SyncClient())

    batch = client.upload_files(
        ("first.txt", "bad.txt", "third.txt"),
        purpose="user_data",
    )

    assert [result.succeeded for result in batch.results] == [True, False, True]
    assert isinstance(batch.results[1].error, RuntimeError)
    assert batch.ok is False
    assert len(batch.succeeded) == 2
    assert len(batch.failed) == 1


def test_vector_store_lifecycle_preserves_raw_resources() -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)

    created = client.create_vector_store(
        name="guides",
        file_ids=("file_1",),
        metadata={"team": "docs"},
        expires_after={"anchor": "last_active_at", "days": 7},
    )
    retrieved = client.retrieve_vector_store("vs_created")
    listed = client.list_vector_stores(limit=2, order="desc")
    updated = client.update_vector_store("vs_created", name="guides-v2")
    deleted = client.delete_vector_store("vs_created")

    assert created.resource is not None
    assert created.resource.id == "vs_created"
    assert created.resource.raw is created.raw
    assert retrieved.id == "vs_created"
    assert [store.id for store in listed] == ["vs_1", "vs_2"]
    assert updated.resource is not None
    assert updated.resource.name == "guides-v2"
    assert deleted.succeeded is True
    assert sdk.vector_stores.created[0]["file_ids"] == ("file_1",)


def test_attach_and_detach_are_distinct_from_file_deletion() -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)
    polling = PollingConfig(poll_interval_ms=250, timeout_seconds=5)

    attached = client.attach_file(
        "vs_1",
        "file_1",
        attributes={"region": "eu"},
        polling=polling,
    )
    detached = client.detach_file("vs_1", "file_1")

    assert attached.succeeded is True
    assert attached.resource is not None
    assert attached.resource.status == "completed"
    call = sdk.vector_stores.files.attach_calls[0]
    assert call["poll_interval_ms"] == 250
    assert call["timeout"] == 5
    assert detached.succeeded is True
    assert sdk.files.deleted == []
    assert sdk.vector_stores.files.detach_calls == [("vs_1", "file_1")]


def test_polling_preserves_timeout_and_rejects_non_terminal_results() -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)
    timeout = TimeoutError("polling timed out")
    sdk.vector_stores.files.error = timeout

    with pytest.raises(TimeoutError) as exc_info:
        client.attach_file("vs_1", "file_1")
    assert exc_info.value is timeout

    sdk.vector_stores.files.error = None
    sdk.vector_stores.files.status = "in_progress"
    with pytest.raises(RuntimeError, match="non-terminal"):
        client.attach_file("vs_1", "file_1")


def test_cleanup_requires_explicit_delete_calls() -> None:
    sdk = SyncClient()
    client = OpenAIRetrievalClient(sdk)

    client.upload_file("guide.txt", purpose="user_data")
    client.create_vector_store(name="guides")

    assert sdk.files.deleted == []
    assert sdk.vector_stores.deleted == []

    assert client.delete_file("file_1").succeeded is True
    assert client.delete_vector_store("vs_created").succeeded is True
    assert sdk.files.deleted == ["file_1"]
    assert sdk.vector_stores.deleted == ["vs_created"]


@pytest.mark.asyncio
async def test_async_client_matches_sync_lifecycle() -> None:
    sdk = AsyncClient()
    client = AsyncOpenAIRetrievalClient(sdk)

    uploaded = await client.upload_file("guide.txt", purpose="user_data")
    created = await client.create_vector_store(name="guides")
    attached = await client.attach_file("vs_created", "file_1")
    listed = await client.list_vector_stores()
    detached = await client.detach_file("vs_created", "file_1")
    deleted_file = await client.delete_file("file_1")
    deleted_store = await client.delete_vector_store("vs_created")

    assert uploaded.resource is not None
    assert uploaded.resource.id == "file_1"
    assert created.resource is not None
    assert created.resource.id == "vs_created"
    assert attached.resource is not None
    assert attached.resource.status == "completed"
    assert listed[0].id == "vs_1"
    assert detached.succeeded is True
    assert deleted_file.succeeded is True
    assert deleted_store.succeeded is True

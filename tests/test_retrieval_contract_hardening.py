"""Regression tests for retrieval lifecycle contract hardening."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest
from openai.resources.vector_stores.files import AsyncFiles, Files

from openai_sdk_helpers.retrieval import (
    AsyncOpenAIRetrievalClient,
    AsyncRetrievalClient,
    AsyncRetrievalLifecycleClient,
    OpenAIRetrievalClient,
    SyncRetrievalClient,
    SyncRetrievalLifecycleClient,
)


class InterruptingSyncClient(OpenAIRetrievalClient):
    """Lifecycle client whose upload simulates process interruption.

    Methods
    -------
    upload_file(source, **kwargs)
        Raise ``KeyboardInterrupt`` to verify propagation.
    """

    def upload_file(self, source: object, **kwargs: Any) -> Any:
        """Raise a process-level interruption."""
        raise KeyboardInterrupt


class CancellingAsyncClient(AsyncOpenAIRetrievalClient):
    """Lifecycle client whose upload simulates task cancellation.

    Methods
    -------
    upload_file(source, **kwargs)
        Raise ``asyncio.CancelledError`` to verify propagation.
    """

    async def upload_file(self, source: object, **kwargs: Any) -> Any:
        """Raise task cancellation."""
        raise asyncio.CancelledError


def _sdk_shape() -> object:
    """Return the minimal client shape accepted by lifecycle constructors."""
    return SimpleNamespace(files=SimpleNamespace(), vector_stores=SimpleNamespace())


def test_sync_batch_does_not_swallow_process_interruptions() -> None:
    """Propagate ``KeyboardInterrupt`` even when ordinary failures continue."""
    client = InterruptingSyncClient(_sdk_shape())

    with pytest.raises(KeyboardInterrupt):
        client.upload_files(("one.txt",), purpose="user_data", continue_on_error=True)


@pytest.mark.asyncio
async def test_async_batch_does_not_swallow_task_cancellation() -> None:
    """Propagate ``CancelledError`` instead of recording it as an item failure."""
    client = CancellingAsyncClient(_sdk_shape())

    with pytest.raises(asyncio.CancelledError):
        await client.upload_files(
            ("one.txt",),
            purpose="user_data",
            continue_on_error=True,
        )


def test_concrete_clients_satisfy_lifecycle_protocols_only() -> None:
    """Keep lifecycle and full-search structural contracts distinct."""
    sync_client = OpenAIRetrievalClient(_sdk_shape())
    async_client = AsyncOpenAIRetrievalClient(_sdk_shape())

    assert isinstance(sync_client, SyncRetrievalLifecycleClient)
    assert isinstance(async_client, AsyncRetrievalLifecycleClient)
    assert not isinstance(sync_client, SyncRetrievalClient)
    assert not isinstance(async_client, AsyncRetrievalClient)


def test_full_protocol_type_hints_resolve_at_runtime() -> None:
    """Keep public vector-store attachment annotations introspectable."""
    sync_hints = get_type_hints(SyncRetrievalClient.attach_file)
    async_hints = get_type_hints(AsyncRetrievalClient.attach_file)

    assert "VectorStoreFileReference" in str(sync_hints["return"])
    assert "VectorStoreFileReference" in str(async_hints["return"])


def test_supported_sdk_polling_helpers_accept_timeout() -> None:
    """Lock the request-timeout passthrough against supported SDK versions."""
    for helper in (Files.create_and_poll, AsyncFiles.create_and_poll):
        parameters = inspect.signature(helper).parameters
        assert "poll_interval_ms" in parameters
        assert "timeout" in parameters

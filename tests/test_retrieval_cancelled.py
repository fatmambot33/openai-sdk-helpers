"""Regression tests for cancelled retrieval polling outcomes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openai_sdk_helpers.retrieval import (
    AsyncOpenAIRetrievalClient,
    OpenAIRetrievalClient,
)


class _SyncVectorStoreFiles:
    """SDK-shaped synchronous vector-store file resource."""

    def create_and_poll(self, **kwargs: Any) -> object:
        """Return a cancelled terminal attachment."""
        return SimpleNamespace(
            id=kwargs["file_id"],
            status="cancelled",
            last_error=None,
        )


class _AsyncVectorStoreFiles:
    """SDK-shaped asynchronous vector-store file resource."""

    async def create_and_poll(self, **kwargs: Any) -> object:
        """Return a cancelled terminal attachment."""
        return SimpleNamespace(
            id=kwargs["file_id"],
            status="cancelled",
            last_error=None,
        )


def test_cancelled_attachment_is_not_successful() -> None:
    """Preserve cancellation status without reporting success."""
    client = OpenAIRetrievalClient(
        SimpleNamespace(
            files=SimpleNamespace(),
            vector_stores=SimpleNamespace(files=_SyncVectorStoreFiles()),
        )
    )

    result = client.attach_file("vs_1", "file_1")

    assert result.succeeded is False
    assert result.resource is not None
    assert result.resource.status == "cancelled"


@pytest.mark.asyncio
async def test_async_cancelled_attachment_is_not_successful() -> None:
    """Match cancellation semantics in the asynchronous client."""
    client = AsyncOpenAIRetrievalClient(
        SimpleNamespace(
            files=SimpleNamespace(),
            vector_stores=SimpleNamespace(files=_AsyncVectorStoreFiles()),
        )
    )

    result = await client.attach_file("vs_1", "file_1")

    assert result.succeeded is False
    assert result.resource is not None
    assert result.resource.status == "cancelled"

"""Full retrieval clients combining lifecycle and direct search."""

from __future__ import annotations

from .client import (
    AsyncOpenAIRetrievalClient as AsyncRetrievalLifecycleClient,
)
from .client import OpenAIRetrievalClient as RetrievalLifecycleClient
from .search import AsyncSearchMixin, SyncSearchMixin


class OpenAIRetrievalClient(SyncSearchMixin, RetrievalLifecycleClient):
    """Synchronous OpenAI retrieval lifecycle and direct search client."""


class AsyncOpenAIRetrievalClient(
    AsyncSearchMixin,
    AsyncRetrievalLifecycleClient,
):
    """Asynchronous OpenAI retrieval lifecycle and direct search client."""


__all__ = ["AsyncOpenAIRetrievalClient", "OpenAIRetrievalClient"]

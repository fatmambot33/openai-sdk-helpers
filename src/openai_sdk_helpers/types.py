"""Common type definitions shared across the SDK."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union

from openai import OpenAI


class SupportsOpenAIClient(Protocol):
    """Protocol describing the subset of the OpenAI client the SDK relies on."""

    api_key: Optional[str]
    vector_stores: Any
    responses: Any
    files: Any


OpenAIClientLike = Union[OpenAI, SupportsOpenAIClient]


__all__ = ["SupportsOpenAIClient", "OpenAIClientLike"]

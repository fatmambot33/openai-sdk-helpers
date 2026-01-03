"""Common type definitions shared across the SDK.

This module defines protocol types and type aliases for OpenAI client
compatibility, enabling flexible client usage throughout the package.

Classes
-------
SupportsOpenAIClient
    Protocol describing the subset of OpenAI client interface used by the SDK.

Type Aliases
------------
OpenAIClient
    Union type accepting OpenAI client or compatible protocol implementations.
"""

from __future__ import annotations

from typing import Any, Protocol

from openai import OpenAI


class SupportsOpenAIClient(Protocol):
    """Protocol describing the subset of the OpenAI client the SDK relies on.

    Defines the minimum interface required for OpenAI client compatibility.
    Custom implementations can satisfy this protocol for testing or
    alternative backends.

    Attributes
    ----------
    api_key : str or None
        API key for authentication.
    vector_stores : Any
        Vector stores management interface.
    responses : Any
        Responses API interface.
    files : Any
        Files management interface.
    """

    api_key: str | None
    vector_stores: Any
    responses: Any
    files: Any


OpenAIClient = OpenAI | SupportsOpenAIClient
"""Type alias for OpenAI client or compatible protocol implementation.

Accepts either the official OpenAI client or any object satisfying the
SupportsOpenAIClient protocol.
"""


__all__ = ["SupportsOpenAIClient", "OpenAIClient"]

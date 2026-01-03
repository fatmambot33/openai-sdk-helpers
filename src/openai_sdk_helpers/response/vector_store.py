"""Vector store attachment utilities for responses.

This module provides functions for attaching named vector stores to response
instances, enabling file search capabilities through the OpenAI API.
"""

from __future__ import annotations

from typing import Any, Sequence

from openai import OpenAI

from ..utils import ensure_list
from .base import BaseResponse


def attach_vector_store(
    response: BaseResponse[Any],
    vector_stores: str | Sequence[str],
    api_key: str | None = None,
) -> list[str]:
    """Attach named vector stores to a response's file_search tool.

    Resolves vector store names to IDs via the OpenAI API and configures
    the response's file_search tool to use them. Creates the file_search
    tool if it doesn't exist, or updates it to include additional stores.

    Parameters
    ----------
    response : BaseResponse[Any]
        Response instance whose tool configuration will be updated.
    vector_stores : str or Sequence[str]
        Single vector store name or sequence of names to attach.
    api_key : str or None, default None
        API key for OpenAI client. If None, uses the response's client.

    Returns
    -------
    list[str]
        Ordered list of vector store IDs attached to the file_search tool.

    Raises
    ------
    ValueError
        If a vector store name cannot be resolved to an ID.
        If no API key is available and the response has no client.

    Examples
    --------
    >>> from openai_sdk_helpers.response import attach_vector_store
    >>> ids = attach_vector_store(response, "knowledge_base")
    >>> ids = attach_vector_store(response, ["docs", "kb"], api_key="sk-...")
    """
    requested_stores = ensure_list(vector_stores)

    client = getattr(response, "_client", None)
    if client is None:
        if api_key is None:
            raise ValueError(
                "OpenAI API key is required to resolve vector store names."
            )
        client = OpenAI(api_key=api_key)

    available_stores = client.vector_stores.list().data
    resolved_ids: list[str] = []

    for store in requested_stores:
        match = next(
            (vs.id for vs in available_stores if vs.name == store),
            None,
        )
        if match is None:
            raise ValueError(f"Vector store '{store}' not found.")
        if match not in resolved_ids:
            resolved_ids.append(match)

    file_search_tool = next(
        (tool for tool in response._tools if tool.get("type") == "file_search"),
        None,
    )

    if file_search_tool is None:
        response._tools.append(
            {"type": "file_search", "vector_store_ids": resolved_ids}
        )
        return resolved_ids

    existing_ids = ensure_list(file_search_tool.get("vector_store_ids", []))
    combined_ids = existing_ids.copy()
    for vector_store_id in resolved_ids:
        if vector_store_id not in combined_ids:
            combined_ids.append(vector_store_id)
    file_search_tool["vector_store_ids"] = combined_ids
    return combined_ids


__all__ = ["attach_vector_store"]

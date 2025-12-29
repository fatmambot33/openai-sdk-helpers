"""Helpers for attaching vector stores to responses."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from openai import OpenAI

from ..utils import ensure_list
from .base import ResponseBase


def attach_vector_store(
    response: ResponseBase[Any],
    vector_stores: str | Sequence[str],
    api_key: Optional[str] = None,
) -> list[str]:
    """Attach vector stores to a response ``file_search`` tool.

    Parameters
    ----------
    response
        Response instance whose tool configuration is updated.
    vector_stores
        Single vector store name or a sequence of names to attach.
    api_key : str, optional
        API key used when the response does not already have a client. Default
        ``None``.

    Returns
    -------
    list[str]
        Ordered list of vector store IDs applied to the ``file_search`` tool.

    Raises
    ------
    ValueError
        If a vector store cannot be resolved or no API key is available when
        required.
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

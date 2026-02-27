"""Helpers for OpenAI websocket-mode connections.

This module provides convenience helpers for the Responses API websocket mode,
including opening persistent connections and building ``response.create``
events for incremental continuation workflows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openai_sdk_helpers.settings import OpenAISettings
from openai_sdk_helpers.types import OpenAIClient


def open_websocket_connection(
    *,
    openai_settings: OpenAISettings | None = None,
    client: OpenAIClient | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra_query: Mapping[str, str] | None = None,
    websocket_connection_options: Mapping[str, Any] | None = None,
) -> Any:
    """Open an OpenAI Responses API websocket connection.

    Use this helper to open websocket-mode sessions while reusing the same
    settings and client lifecycle conventions used by ``openai-sdk-helpers``.

    Parameters
    ----------
    openai_settings : OpenAISettings or None, optional
        Settings used to build a temporary client when ``client`` is not
        provided, by default None.
    client : OpenAIClient or None, optional
        Existing OpenAI client instance. If provided, it takes precedence over
        ``openai_settings``, by default None.
    extra_headers : Mapping[str, str] or None, optional
        Additional HTTP headers for the websocket handshake, by default None.
    extra_query : Mapping[str, str] or None, optional
        Additional query string parameters for the websocket endpoint, by
        default None.
    websocket_connection_options : Mapping[str, Any] or None, optional
        Low-level websocket transport options passed through to the OpenAI SDK,
        by default None.

    Returns
    -------
    Any
        OpenAI SDK responses websocket connection manager. Use it as a context
        manager, send events such as ``response.create``, and iterate over
        server events.

    Raises
    ------
    ValueError
        If neither ``client`` nor ``openai_settings`` is provided.
    """
    if client is None:
        if openai_settings is None:
            raise ValueError(
                "Provide either `client` or `openai_settings` to open a websocket "
                "connection."
            )
        client = openai_settings.create_client()

    connect_kwargs: dict[str, Any] = {
        "extra_headers": dict(extra_headers or {}),
        "extra_query": dict(extra_query or {}),
        "websocket_connection_options": dict(websocket_connection_options or {}),
    }

    return client.responses.connect(**connect_kwargs)


def build_response_create_event(
    *,
    model: str,
    input_items: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    store: bool | None = None,
    previous_response_id: str | None = None,
    generate: bool | None = None,
    context_management: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a ``response.create`` websocket event payload.

    Parameters
    ----------
    model : str
        Model name for response generation.
    input_items : Sequence[Mapping[str, Any]]
        New input items for this turn. Pass only incremental items when
        continuing a chain with ``previous_response_id``.
    tools : Sequence[Mapping[str, Any]] or None, optional
        Tool definitions forwarded to the model, by default None.
    store : bool or None, optional
        Whether to persist response state. Use ``False`` for store=false
        patterns, by default None.
    previous_response_id : str or None, optional
        Prior response identifier for continuation, by default None.
    generate : bool or None, optional
        Set to ``False`` to warm up request state without generating output,
        by default None.
    context_management : Sequence[Mapping[str, Any]] or None, optional
        Context compaction settings forwarded to the API when provided,
        by default None.

    Returns
    -------
    dict[str, Any]
        Event body suitable for ``connection.send(...)``.

    Examples
    --------
    >>> event = build_response_create_event(
    ...     model="gpt-5.2",
    ...     store=False,
    ...     input_items=[
    ...         {
    ...             "type": "message",
    ...             "role": "user",
    ...             "content": [{"type": "input_text", "text": "Find fizz_buzz()"}],
    ...         }
    ...     ],
    ... )
    >>> event["type"]
    'response.create'
    """
    event: dict[str, Any] = {
        "type": "response.create",
        "model": model,
        "input": list(input_items),
        "tools": list(tools or []),
    }
    if store is not None:
        event["store"] = store
    if previous_response_id is not None:
        event["previous_response_id"] = previous_response_id
    if generate is not None:
        event["generate"] = generate
    if context_management is not None:
        event["context_management"] = list(context_management)
    return event


def send_response_create(
    connection: Any,
    *,
    model: str,
    input_items: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    store: bool | None = None,
    previous_response_id: str | None = None,
    generate: bool | None = None,
    context_management: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and send a ``response.create`` event to a websocket connection.

    Parameters
    ----------
    connection : Any
        Active websocket connection returned by ``open_websocket_connection``.
    model : str
        Model name for response generation.
    input_items : Sequence[Mapping[str, Any]]
        New input items for this turn.
    tools : Sequence[Mapping[str, Any]] or None, optional
        Tool definitions forwarded to the model, by default None.
    store : bool or None, optional
        Whether to persist response state, by default None.
    previous_response_id : str or None, optional
        Prior response identifier for continuation, by default None.
    generate : bool or None, optional
        Set to ``False`` to warm up request state, by default None.
    context_management : Sequence[Mapping[str, Any]] or None, optional
        Context compaction settings forwarded to the API, by default None.

    Returns
    -------
    dict[str, Any]
        Event payload that was sent.
    """
    event = build_response_create_event(
        model=model,
        input_items=input_items,
        tools=tools,
        store=store,
        previous_response_id=previous_response_id,
        generate=generate,
        context_management=context_management,
    )
    connection.send(event)
    return event


__all__ = [
    "open_websocket_connection",
    "build_response_create_event",
    "send_response_create",
]

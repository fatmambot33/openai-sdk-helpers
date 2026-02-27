"""Tests for websocket-mode helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openai_sdk_helpers.response.websocket import (
    build_response_create_event,
    open_websocket_connection,
    send_response_create,
)


def test_open_websocket_connection_uses_explicit_client() -> None:
    """Use provided client and forward websocket options to connect."""
    connect = MagicMock(return_value="connection")
    client = SimpleNamespace(responses=SimpleNamespace(connect=connect))

    result = open_websocket_connection(
        client=client,
        extra_headers={"x-test": "1"},
        extra_query={"trace": "true"},
        websocket_connection_options={"max_size": 2**20},
    )

    assert result == "connection"
    connect.assert_called_once_with(
        extra_headers={"x-test": "1"},
        extra_query={"trace": "true"},
        websocket_connection_options={"max_size": 2**20},
    )


def test_open_websocket_connection_builds_client_from_settings() -> None:
    """Build a client from settings when no explicit client is provided."""
    connect = MagicMock(return_value="connection")
    client = SimpleNamespace(responses=SimpleNamespace(connect=connect))
    settings = MagicMock()
    settings.create_client.return_value = client

    result = open_websocket_connection(openai_settings=settings)

    assert result == "connection"
    settings.create_client.assert_called_once_with()
    connect.assert_called_once_with(
        extra_headers={},
        extra_query={},
        websocket_connection_options={},
    )


def test_open_websocket_connection_requires_client_or_settings() -> None:
    """Raise ValueError when neither client nor settings are provided."""
    with pytest.raises(
        ValueError, match="Provide either `client` or `openai_settings`"
    ):
        open_websocket_connection()


def test_build_response_create_event_with_continuation_fields() -> None:
    """Build continuation payload with previous response and warmup options."""
    event = build_response_create_event(
        model="gpt-5.2",
        store=False,
        previous_response_id="resp_123",
        generate=False,
        input_items=[
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "tool result",
            }
        ],
        tools=[{"type": "function", "name": "demo", "parameters": {}}],
    )

    assert event == {
        "type": "response.create",
        "model": "gpt-5.2",
        "store": False,
        "previous_response_id": "resp_123",
        "generate": False,
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "tool result",
            }
        ],
        "tools": [{"type": "function", "name": "demo", "parameters": {}}],
    }


def test_send_response_create_sends_built_event() -> None:
    """Send built response.create payload through active websocket connection."""
    connection = MagicMock()

    sent = send_response_create(
        connection,
        model="gpt-5.2",
        store=False,
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Find fizz_buzz()"}],
            }
        ],
    )

    connection.send.assert_called_once_with(sent)
    assert sent["type"] == "response.create"
    assert sent["model"] == "gpt-5.2"

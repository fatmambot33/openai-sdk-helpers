"""Tests for thin Realtime API helpers without live network access."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from openai_sdk_helpers.realtime import (
    AsyncRealtimeClient,
    RealtimeClient,
    RealtimeUnavailableError,
    normalize_realtime_event,
    realtime_available,
)
from openai_sdk_helpers.runtime import OperationContext, OperationPhase


class _SyncSessions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = object()

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.result


class _AsyncSessions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = object()

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.result


class _SyncConnection:
    def __init__(self) -> None:
        self.session = SimpleNamespace(update=self._update_session)
        self.response = SimpleNamespace(
            cancel=self._cancel_response,
            create=self._create_response,
        )
        self.conversation = SimpleNamespace(
            item=SimpleNamespace(create=self._create_item),
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _update_session(self, **kwargs: Any) -> None:
        self.calls.append(("session.update", kwargs))

    def _cancel_response(self, **kwargs: Any) -> None:
        self.calls.append(("response.cancel", kwargs))

    def _create_response(self, **kwargs: Any) -> None:
        self.calls.append(("response.create", kwargs))

    def _create_item(self, **kwargs: Any) -> None:
        self.calls.append(("conversation.item.create", kwargs))


class _AsyncConnection:
    def __init__(self) -> None:
        self.session = SimpleNamespace(update=self._update_session)
        self.response = SimpleNamespace(
            cancel=self._cancel_response,
            create=self._create_response,
        )
        self.conversation = SimpleNamespace(
            item=SimpleNamespace(create=self._create_item),
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def _update_session(self, **kwargs: Any) -> None:
        self.calls.append(("session.update", kwargs))

    async def _cancel_response(self, **kwargs: Any) -> None:
        self.calls.append(("response.cancel", kwargs))

    async def _create_response(self, **kwargs: Any) -> None:
        self.calls.append(("response.create", kwargs))

    async def _create_item(self, **kwargs: Any) -> None:
        self.calls.append(("conversation.item.create", kwargs))


class _AsyncEvents:
    def __init__(self, events: tuple[object, ...]) -> None:
        self._events = events

    def __aiter__(self) -> AsyncIterator[object]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event


def _client(client_secrets: object, connect: Any) -> SimpleNamespace:
    return SimpleNamespace(
        realtime=SimpleNamespace(client_secrets=client_secrets, connect=connect)
    )


def test_capability_detection_and_error_are_actionable() -> None:
    """Detect the API structurally and fail before any request."""
    supported = _client(_SyncSessions(), lambda **kwargs: kwargs)
    unsupported = SimpleNamespace()

    assert realtime_available(supported)
    assert not realtime_available(unsupported)

    with pytest.raises(RealtimeUnavailableError) as error:
        RealtimeClient(unsupported)  # type: ignore[arg-type]

    assert "openai[realtime]>=3.41.0" in str(error.value)
    assert error.value.context == {
        "feature": "realtime",
        "minimum_openai_version": "3.41.0",
    }


def test_sync_lifecycle_connection_and_explicit_events() -> None:
    """Forward sync session lifecycle and event helpers to raw SDK resources."""
    sessions = _SyncSessions()
    connect_calls: list[dict[str, Any]] = []
    sdk_client = _client(
        sessions, lambda **kwargs: connect_calls.append(kwargs) or "manager"
    )
    helper = RealtimeClient(sdk_client)  # type: ignore[arg-type]
    observed = []

    result = helper.create_client_secret(
        session={
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["text"],
        },
        operation_context=OperationContext(
            "realtime.secret", observers=(observed.append,)
        ),
    )

    assert helper.sdk_client is sdk_client
    assert helper.realtime is sdk_client.realtime
    assert helper.client_secrets is sessions
    assert result is sessions.result
    assert sessions.calls == [
        {
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["text"],
            }
        }
    ]
    assert [event.phase for event in observed] == [
        OperationPhase.START,
        OperationPhase.SUCCESS,
    ]

    assert (
        helper.connect(
            model="gpt-realtime",
            extra_headers={"x-test": "1"},
            extra_query={"trace": "true"},
            websocket_connection_options={"max_size": 1024},
        )
        == "manager"
    )
    assert connect_calls == [
        {
            "model": "gpt-realtime",
            "extra_headers": {"x-test": "1"},
            "extra_query": {"trace": "true"},
            "websocket_connection_options": {"max_size": 1024},
        }
    ]

    connection = _SyncConnection()
    helper.update_session(
        connection,
        session={"type": "realtime", "instructions": "Be concise."},
        event_id="event_session",
    )
    helper.cancel_response(connection, response_id="resp_123", event_id="event_cancel")
    helper.submit_tool_result(
        connection,
        call_id="call_123",
        output='{"status":"ok"}',
        event_id="event_tool",
    )
    helper.submit_tool_result(
        connection,
        call_id="call_456",
        output="done",
        create_response=False,
    )

    assert connection.calls == [
        (
            "session.update",
            {
                "session": {
                    "type": "realtime",
                    "instructions": "Be concise.",
                },
                "event_id": "event_session",
            },
        ),
        (
            "response.cancel",
            {"response_id": "resp_123", "event_id": "event_cancel"},
        ),
        (
            "conversation.item.create",
            {
                "item": {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": '{"status":"ok"}',
                },
                "event_id": "event_tool",
            },
        ),
        ("response.create", {}),
        (
            "conversation.item.create",
            {
                "item": {
                    "type": "function_call_output",
                    "call_id": "call_456",
                    "output": "done",
                }
            },
        ),
    ]


def test_normalized_events_preserve_raw_sdk_events() -> None:
    """Normalize only the event envelope and retain native event objects."""
    raw_model = SimpleNamespace(type="response.done", response=object())
    raw_mapping = {"type": "session.updated", "session": {"model": "gpt-realtime"}}
    helper = RealtimeClient(_client(_SyncSessions(), lambda **kwargs: kwargs))  # type: ignore[arg-type]

    direct = normalize_realtime_event(raw_model)
    events = list(helper.iter_events((raw_model, raw_mapping)))

    assert direct.type == "response.done"
    assert direct.raw is raw_model
    assert [event.type for event in events] == ["response.done", "session.updated"]
    assert events[0].raw is raw_model
    assert events[1].raw is raw_mapping

    with pytest.raises(ValueError, match="non-empty string `type`"):
        normalize_realtime_event({"type": ""})


@pytest.mark.asyncio
async def test_async_facade_matches_sync_semantics() -> None:
    """Forward asynchronous lifecycle and event helpers without a network call."""
    sessions = _AsyncSessions()
    connect_calls: list[dict[str, Any]] = []
    sdk_client = _client(
        sessions, lambda **kwargs: connect_calls.append(kwargs) or "manager"
    )
    helper = AsyncRealtimeClient(sdk_client)  # type: ignore[arg-type]
    connection = _AsyncConnection()

    assert (
        await helper.create_client_secret(
            session={
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["text"],
            }
        )
        is sessions.result
    )
    assert helper.connect(model="gpt-realtime") == "manager"
    await helper.update_session(
        connection,
        session={"type": "realtime", "output_modalities": ["text"]},
    )
    await helper.cancel_response(connection)
    await helper.submit_tool_result(connection, call_id="call_123", output="ok")

    events = [
        event
        async for event in helper.iter_events(
            _AsyncEvents(({"type": "response.done"},))
        )
    ]

    assert sessions.calls == [
        {
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["text"],
            }
        }
    ]
    assert connect_calls == [
        {
            "model": "gpt-realtime",
            "extra_headers": {},
            "extra_query": {},
            "websocket_connection_options": {},
        }
    ]
    assert connection.calls == [
        (
            "session.update",
            {
                "session": {
                    "type": "realtime",
                    "output_modalities": ["text"],
                }
            },
        ),
        ("response.cancel", {}),
        (
            "conversation.item.create",
            {
                "item": {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "ok",
                }
            },
        ),
        ("response.create", {}),
    ]
    assert events[0].type == "response.done"

"""Credential-free smoke example for the Realtime API helper surface."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from openai_sdk_helpers.realtime import RealtimeClient, realtime_available


class _FakeSessions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(client_secret=SimpleNamespace(value="not-a-real-secret"))


class _FakeConnection:
    def __init__(self) -> None:
        self.session = SimpleNamespace(update=self._session_update)
        self.response = SimpleNamespace(
            cancel=self._response_cancel, create=self._response_create
        )
        self.conversation = SimpleNamespace(
            item=SimpleNamespace(create=self._item_create)
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _session_update(self, **kwargs: Any) -> None:
        self.calls.append(("session.update", kwargs))

    def _response_cancel(self, **kwargs: Any) -> None:
        self.calls.append(("response.cancel", kwargs))

    def _response_create(self, **kwargs: Any) -> None:
        self.calls.append(("response.create", kwargs))

    def _item_create(self, **kwargs: Any) -> None:
        self.calls.append(("conversation.item.create", kwargs))


sessions = _FakeSessions()
realtime = SimpleNamespace(client_secrets=sessions, connect=lambda **kwargs: kwargs)
sdk_client = SimpleNamespace(realtime=realtime)

assert realtime_available(sdk_client)

client = RealtimeClient(sdk_client)  # type: ignore[arg-type]
secret = client.create_client_secret(
    session={"type": "realtime", "model": "gpt-realtime", "modalities": ["text"]}
)
assert secret.client_secret.value == "not-a-real-secret"
assert sessions.calls == [
    {
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "modalities": ["text"],
        }
    }
]

connection = _FakeConnection()
client.update_session(connection, session={"instructions": "Be concise."})
client.submit_tool_result(connection, call_id="call_123", output='{"status":"ok"}')
assert connection.calls[-1] == ("response.create", {})

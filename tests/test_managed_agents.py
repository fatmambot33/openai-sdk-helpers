"""Tests for the managed Agents API facade."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openai_sdk_helpers.managed_agents import (
    AsyncManagedAgentsClient,
    ManagedAgentsClient,
    ManagedAgentsUnavailableError,
    managed_agents_available,
)
from openai_sdk_helpers.runtime import OperationContext, OperationPhase


class _SyncEvents:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create(self, session_id: str, **kwargs: Any) -> None:
        self.calls.append((session_id, kwargs))


class _SyncSessions:
    def __init__(self) -> None:
        self.events = _SyncEvents()
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.created = object()
        self.retrieved = object()
        self.updated = object()
        self.page = object()
        self.deleted = object()

    def create(self, **kwargs: Any) -> object:
        self.calls.append(("create", (), kwargs))
        return self.created

    def retrieve(self, *args: Any, **kwargs: Any) -> object:
        self.calls.append(("retrieve", args, kwargs))
        return self.retrieved

    def update(self, *args: Any, **kwargs: Any) -> object:
        self.calls.append(("update", args, kwargs))
        return self.updated

    def list(self, **kwargs: Any) -> object:
        self.calls.append(("list", (), kwargs))
        return self.page

    def delete(self, *args: Any, **kwargs: Any) -> object:
        self.calls.append(("delete", args, kwargs))
        return self.deleted


class _AsyncEvents:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def create(self, session_id: str, **kwargs: Any) -> None:
        self.calls.append((session_id, kwargs))


class _AsyncSessions:
    def __init__(self) -> None:
        self.events = _AsyncEvents()
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.created = object()
        self.retrieved = object()
        self.updated = object()
        self.page = object()
        self.deleted = object()

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(("create", (), kwargs))
        return self.created

    async def retrieve(self, *args: Any, **kwargs: Any) -> object:
        self.calls.append(("retrieve", args, kwargs))
        return self.retrieved

    async def update(self, *args: Any, **kwargs: Any) -> object:
        self.calls.append(("update", args, kwargs))
        return self.updated

    def list(self, **kwargs: Any) -> object:
        self.calls.append(("list", (), kwargs))
        return self.page

    async def delete(self, *args: Any, **kwargs: Any) -> object:
        self.calls.append(("delete", args, kwargs))
        return self.deleted


def _client_with_sessions(sessions: object) -> SimpleNamespace:
    agents = SimpleNamespace(sessions=sessions)
    return SimpleNamespace(beta=SimpleNamespace(agents=agents))


def test_capability_detection_and_error_are_actionable() -> None:
    """Detect the feature structurally and fail before any SDK request."""
    supported = _client_with_sessions(_SyncSessions())
    unsupported = SimpleNamespace(beta=SimpleNamespace())

    assert managed_agents_available(supported)
    assert not managed_agents_available(unsupported)

    with pytest.raises(ManagedAgentsUnavailableError) as error:
        ManagedAgentsClient(unsupported)  # type: ignore[arg-type]

    assert "openai>=3.13.0" in str(error.value)
    assert error.value.context["feature"] == "managed_agents"
    assert error.value.context["minimum_openai_version"] == "3.13.0"


def test_sync_facade_preserves_resources_results_and_observability() -> None:
    """Forward sync lifecycle operations without normalizing SDK results."""
    sessions = _SyncSessions()
    sdk_client = _client_with_sessions(sessions)
    helper = ManagedAgentsClient(sdk_client)  # type: ignore[arg-type]
    observed = []
    context = OperationContext("managed_agents.create", observers=(observed.append,))

    created = helper.create_session(
        environment={"type": "openai_hosted"},
        agent_id="agent_123",
        input="Inspect the repository",
        metadata={"source": "test"},
        vault_ids=("vault_1",),
        operation_context=context,
    )

    assert helper.sdk_client is sdk_client
    assert helper.agents is sdk_client.beta.agents
    assert helper.sessions is sessions
    assert created is sessions.created
    assert sessions.calls[0] == (
        "create",
        (),
        {
            "environment": {"type": "openai_hosted"},
            "agent_id": "agent_123",
            "input": "Inspect the repository",
            "metadata": {"source": "test"},
            "vault_ids": ["vault_1"],
        },
    )
    assert [event.phase for event in observed] == [
        OperationPhase.START,
        OperationPhase.SUCCESS,
    ]
    assert observed[-1].result is sessions.created

    assert helper.retrieve_session(" session_1 ") is sessions.retrieved
    assert helper.update_session_metadata(
        "session_1", metadata=None
    ) is sessions.updated
    assert helper.list_sessions(
        after="session_0", agent_id="agent_123", limit=10, order="asc"
    ) is sessions.page
    assert helper.delete_session("session_1") is sessions.deleted

    events = (
        {
            "type": "agent.session.input.message",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Continue."}],
                }
            ],
        },
    )
    helper.submit_events(
        "session_1",
        events=events,
        idempotency_key="idem_1",
    )
    assert sessions.events.calls == [
        (
            "session_1",
            {
                "events": events,
                "idempotency_key": "idem_1",
            },
        )
    ]


def test_sync_facade_rejects_empty_session_id_before_sdk_call() -> None:
    """Reject empty identifiers without touching the underlying SDK resource."""
    sessions = _SyncSessions()
    helper = ManagedAgentsClient(_client_with_sessions(sessions))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="session_id must not be empty"):
        helper.retrieve_session("   ")

    assert sessions.calls == []


@pytest.mark.asyncio
async def test_async_facade_preserves_results_and_event_submission() -> None:
    """Provide async parity while returning original SDK objects."""
    sessions = _AsyncSessions()
    sdk_client = _client_with_sessions(sessions)
    helper = AsyncManagedAgentsClient(sdk_client)  # type: ignore[arg-type]
    observed = []
    context = OperationContext(
        "managed_agents.retrieve",
        observers=(observed.append,),
    )

    assert await helper.create_session(
        environment={"type": "openai_hosted"},
        agent_id="agent_123",
    ) is sessions.created
    assert await helper.retrieve_session(
        "session_1", operation_context=context
    ) is sessions.retrieved
    assert await helper.update_session_metadata(
        "session_1", metadata={"stage": "test"}
    ) is sessions.updated
    assert helper.list_sessions(limit=5, order="desc") is sessions.page
    assert await helper.delete_session("session_1") is sessions.deleted

    events = ({"type": "agent.session.input.cancel"},)
    await helper.submit_events("session_1", events=events)

    assert helper.sdk_client is sdk_client
    assert helper.sessions is sessions
    assert [event.phase for event in observed] == [
        OperationPhase.START,
        OperationPhase.SUCCESS,
    ]
    assert observed[-1].result is sessions.retrieved
    assert sessions.events.calls == [("session_1", {"events": events})]

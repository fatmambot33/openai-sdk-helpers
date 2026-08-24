"""Tests for explicit conversation state and persistence semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openai_sdk_helpers.agent import runner as agent_runner
from openai_sdk_helpers.response import ResponseMessages
from openai_sdk_helpers.state import (
    AgentRunState,
    ConversationStateMode,
    LocalMessageStore,
    ResponseContinuation,
    resolve_agent_state,
)


def test_response_continuation_is_stateless_by_default() -> None:
    continuation = ResponseContinuation()

    request = {"model": "example-model", "input": "hello"}
    resolved = continuation.apply(request)

    assert continuation.mode is ConversationStateMode.STATELESS
    assert resolved == request
    assert resolved is not request


def test_response_continuation_preserves_identifiers() -> None:
    previous = ResponseContinuation(previous_response_id=" resp_123 ")
    conversation = ResponseContinuation(conversation_id=" conv_123 ")

    assert previous.mode is ConversationStateMode.PREVIOUS_RESPONSE
    assert previous.previous_response_id == "resp_123"
    assert previous.apply({"model": "example"})["previous_response_id"] == "resp_123"
    assert conversation.mode is ConversationStateMode.CONVERSATION
    assert conversation.conversation_id == "conv_123"
    assert conversation.apply({"model": "example"})["conversation"] == "conv_123"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"previous_response_id": "resp_1", "conversation_id": "conv_1"},
        {"previous_response_id": " "},
        {"conversation_id": " "},
    ],
)
def test_response_continuation_rejects_ambiguous_values(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        ResponseContinuation(**kwargs)


def test_response_continuation_rejects_conflicting_request_kwargs() -> None:
    continuation = ResponseContinuation(previous_response_id="resp_2")

    with pytest.raises(ValueError):
        continuation.apply({"conversation": "conv_1"})
    with pytest.raises(ValueError):
        continuation.apply({"previous_response_id": "resp_1"})


def test_agent_state_modes_preserve_underlying_session() -> None:
    session = object()
    session_state = AgentRunState(session=session)
    chained_state = AgentRunState(
        previous_response_id="resp_1",
        auto_previous_response_id=True,
    )

    assert session_state.mode is ConversationStateMode.AGENT_SESSION
    assert session_state.session is session
    assert chained_state.mode is ConversationStateMode.PREVIOUS_RESPONSE
    assert chained_state.previous_response_id == "resp_1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session": object(), "previous_response_id": "resp_1"},
        {"session": object(), "conversation_id": "conv_1"},
        {"session": object(), "auto_previous_response_id": True},
        {"conversation_id": "conv_1", "previous_response_id": "resp_1"},
        {"conversation_id": "conv_1", "auto_previous_response_id": True},
    ],
)
def test_agent_state_rejects_mixed_ownership(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AgentRunState(**kwargs)


def test_legacy_session_shorthand_remains_supported() -> None:
    session = object()

    resolved = resolve_agent_state(state=None, session=session)

    assert resolved.session is session
    with pytest.raises(ValueError, match="either state or session"):
        resolve_agent_state(state=AgentRunState(), session=session)


@pytest.mark.asyncio
async def test_agent_runner_forwards_server_managed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    result = object()

    async def fake_run(*args: Any, **kwargs: Any) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return result

    monkeypatch.setattr(agent_runner.Runner, "run", fake_run)
    state = AgentRunState(
        previous_response_id="resp_123",
        auto_previous_response_id=True,
    )

    returned = await agent_runner.run_async(object(), "hello", state=state)

    assert returned is result
    assert captured["kwargs"]["session"] is None
    assert captured["kwargs"]["previous_response_id"] == "resp_123"
    assert captured["kwargs"]["auto_previous_response_id"] is True
    assert "conversation_id" not in captured["kwargs"]


@pytest.mark.asyncio
async def test_agent_runner_rejects_mixed_forms_before_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_run(*_: Any, **__: Any) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(agent_runner.Runner, "run", fake_run)

    with pytest.raises(ValueError, match="either state or session"):
        await agent_runner.run_async(
            object(),
            "hello",
            session=object(),
            state=AgentRunState(),
        )

    assert called is False


def test_local_message_store_close_without_save_writes_nothing(
    tmp_path: Path,
) -> None:
    store = LocalMessageStore(tmp_path / "history.json")

    result = store.close(ResponseMessages(), save=False)

    assert result is None
    assert store.exists is False


def test_local_message_store_save_resume_clear_and_delete(tmp_path: Path) -> None:
    store = LocalMessageStore(tmp_path / "nested" / "history.json")
    messages = ResponseMessages()

    saved_path = store.close(messages, save=True)
    resumed = store.resume()

    assert saved_path == (tmp_path / "nested" / "history.json").resolve()
    assert store.exists is True
    assert resumed.messages == []

    cleared = store.clear()
    assert cleared.messages == []
    assert store.resume().messages == []

    assert store.delete() is True
    assert store.exists is False
    assert store.delete() is False
    with pytest.raises(FileNotFoundError):
        store.delete(missing_ok=False)


def test_local_message_store_requires_messages_for_save(tmp_path: Path) -> None:
    store = LocalMessageStore(tmp_path / "history.json")

    with pytest.raises(ValueError, match="messages are required"):
        store.close(save=True)

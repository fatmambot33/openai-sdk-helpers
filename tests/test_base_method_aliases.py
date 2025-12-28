from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent.base import AgentBase
from openai_sdk_helpers.response.base import ResponseBase


class _StubAgentBase(AgentBase):
    """Minimal AgentBase subclass for testing async aliases."""

    def __init__(self) -> None:
        # Bypass the parent initializer by setting the required attributes.
        self._output_type = str
        self._run_context_wrapper = None
        self.agent_name = "stub"
        self.description = ""
        self.model = "model"
        self._tools = None
        self._model_settings = None
        self._template = MagicMock(render=MagicMock(return_value=""))

    def get_agent(self) -> Any:  # pragma: no cover - mocked in tests
        return MagicMock()


class _StubResponseBase(ResponseBase[Any]):
    """Minimal ResponseBase subclass for alias testing."""

    def __init__(self) -> None:
        # Avoid base initialization; only attributes used in tests are set.
        self._model = "model"
        self._output_structure = None
        self.messages = MagicMock()


@patch("openai_sdk_helpers.agent.base._run_agent_streamed")
@patch("openai_sdk_helpers.agent.base._run_agent", new_callable=AsyncMock)
@patch("openai_sdk_helpers.agent.base._run_agent_sync")
def test_agent_base_run_aliases(
    mock_run_agent_sync: MagicMock,
    mock_run_agent: AsyncMock,
    mock_run_agent_streamed: MagicMock,
) -> None:
    """Ensure AgentBase convenience helpers call the private runners directly."""

    mock_run_agent.return_value = "async-result"
    mock_run_agent_sync.return_value = MagicMock(
        final_output_as=lambda *_: "sync-result"
    )
    mock_run_agent_streamed.return_value = MagicMock(
        final_output_as=lambda *_: "stream-result"
    )
    agent = _StubAgentBase()

    result_run = agent.run(agent_input="hello")
    result_run_async = asyncio.run(agent.run_async(agent_input="hello"))
    result_stream = agent.run_streamed(agent_input="hello", output_type=str)

    assert result_run == "sync-result"
    assert result_run_async == "async-result"
    assert result_stream == "stream-result"
    mock_run_agent_sync.assert_called_once()
    assert mock_run_agent.await_count == 1
    mock_run_agent_streamed.assert_called_once()


@patch.object(_StubResponseBase, "run_response", return_value="sync-result")
@patch.object(_StubResponseBase, "run_response_async", new_callable=AsyncMock)
def test_response_base_run_aliases(
    mock_run_response_async: AsyncMock, mock_run_response: MagicMock
) -> None:
    """Validate ResponseBase exposes run, run_async, and run_streamed aliases."""

    mock_run_response_async.return_value = "async-result"
    response = _StubResponseBase()

    assert response.run(content="hello") == "sync-result"
    assert asyncio.run(response.run_async(content="hello")) == "async-result"
    assert response.run_streamed(content="hello") == "async-result"
    mock_run_response.assert_called_once_with(content="hello", attachments=None)
    assert mock_run_response_async.await_count == 2

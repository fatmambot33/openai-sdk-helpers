"""Tests for the agent runner convenience functions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent import runner


@pytest.fixture
def mock_agent():
    """Return a mock agent."""
    return MagicMock()


@patch("openai_sdk_helpers.agent.runner.Runner.run", new_callable=AsyncMock)
def test_run_async_returns_coroutine(mock_runner_run, mock_agent):
    """Test that run_async returns a coroutine."""
    mock_runner_run.return_value = MagicMock()
    coro = runner.run_async(mock_agent, input="test_input")
    assert asyncio.iscoroutine(coro)
    asyncio.run(coro)


@patch("openai_sdk_helpers.agent.runner.run_coroutine_with_fallback")
@patch("openai_sdk_helpers.agent.runner.Runner.run", new_callable=AsyncMock)
def test_run_sync(mock_runner_run, mock_run_coroutine, mock_agent):
    """Test the run_sync function."""
    mock_result = MagicMock()
    mock_run_coroutine.return_value = mock_result

    runner.run_sync(mock_agent, input="test_input")

    mock_runner_run.assert_called_once_with(
        mock_agent, "test_input", context=None, session=None
    )
    assert mock_run_coroutine.called


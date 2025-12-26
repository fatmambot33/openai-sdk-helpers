"""Tests for the agent runner convenience functions."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from corai.agent import runner


@pytest.fixture
def mock_agent():
    """Return a mock agent."""
    return MagicMock()


@patch("corai.agent.runner._run_agent")
def test_run_returns_coroutine(mock_run_agent, mock_agent):
    """Test that run() returns a coroutine."""
    coro = runner.run(mock_agent, "test_input")
    assert asyncio.iscoroutine(coro)


@patch("corai.agent.runner._run_agent_sync")
def test_run_sync(mock_run_agent_sync, mock_agent):
    """Test the run_sync function."""
    runner.run_sync(mock_agent, "test_input")
    mock_run_agent_sync.assert_called_once_with(
        agent=mock_agent,
        agent_input="test_input",
        agent_context=None,
    )


@patch("corai.agent.runner._run_agent_streamed")
def test_run_streamed(mock_run_agent_streamed, mock_agent):
    """Test the run_streamed function."""
    runner.run_streamed(mock_agent, "test_input")
    mock_run_agent_streamed.assert_called_once_with(
        agent=mock_agent,
        agent_input="test_input",
        context=None,
    )

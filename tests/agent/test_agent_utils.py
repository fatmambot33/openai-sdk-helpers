"""Tests for the agent utility functions."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from openai_sdk_helpers.agent import utils


async def sample_coro():
    """A sample coroutine for testing."""
    return "test result"


def test_run_coro_sync():
    """Test the run_coro_sync function."""
    result = utils.run_coroutine_agent_sync(sample_coro())
    assert result == "test result"


@patch("asyncio.get_running_loop")
def test_run_coro_sync_with_running_loop(mock_get_running_loop):
    """Test the run_coro_sync function when an event loop is running."""
    mock_loop = MagicMock()
    mock_loop.is_running.return_value = True
    mock_get_running_loop.return_value = mock_loop

    result = utils.run_coroutine_agent_sync(sample_coro())
    assert result == "test result"

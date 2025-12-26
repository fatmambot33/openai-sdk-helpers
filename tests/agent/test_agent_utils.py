"""Tests for the agent utility functions."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from corai.agent import utils


async def sample_coro():
    """A sample coroutine for testing."""
    return "test result"


def test_run_coro_sync():
    """Test the run_coro_sync function."""
    result = utils.run_coro_sync(sample_coro())
    assert result == "test result"


@patch("asyncio.get_running_loop")
def test_run_coro_sync_with_running_loop(mock_get_running_loop):
    """Test the run_coro_sync function when an event loop is running."""
    mock_get_running_loop.return_value.is_running.return_value = True
    result = utils.run_coro_sync(sample_coro())
    assert result == "test result"

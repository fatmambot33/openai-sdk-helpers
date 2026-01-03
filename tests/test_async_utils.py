"""Tests for async utilities module."""

import asyncio

import pytest

from openai_sdk_helpers.async_utils import (
    run_coroutine_thread_safe,
    run_coroutine_with_fallback,
)
from openai_sdk_helpers.errors import AsyncExecutionError


class TestAsyncUtils:
    """Test async/sync bridge utilities."""

    @pytest.mark.asyncio
    async def test_run_coroutine_with_fallback_async(self) -> None:
        """Should run coroutine when no event loop is running."""

        async def sample_coro() -> str:
            return "result"

        result = run_coroutine_with_fallback(sample_coro())
        assert result == "result"

    def test_run_coroutine_with_fallback_sync(self) -> None:
        """Should run coroutine in thread when event loop is already running."""

        async def sample_coro() -> int:
            await asyncio.sleep(0.01)
            return 42

        result = run_coroutine_with_fallback(sample_coro())
        assert result == 42

    def test_run_coroutine_thread_safe_success(self) -> None:
        """Should successfully run coroutine in thread."""

        async def sample_coro() -> str:
            await asyncio.sleep(0.01)
            return "success"

        result = run_coroutine_thread_safe(sample_coro())
        assert result == "success"

    def test_run_coroutine_thread_safe_exception(self) -> None:
        """Should propagate exceptions from coroutine."""

        async def failing_coro() -> None:
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_coroutine_thread_safe(failing_coro())

    def test_run_coroutine_thread_safe_timeout(self) -> None:
        """Should raise AsyncExecutionError on timeout."""

        async def slow_coro() -> None:
            await asyncio.sleep(10)

        with pytest.raises(AsyncExecutionError, match="timed out"):
            run_coroutine_thread_safe(slow_coro(), timeout=0.1)

    def test_run_coroutine_thread_safe_preserves_result_type(self) -> None:
        """Should preserve result type through thread."""

        async def returns_dict() -> dict[str, int]:
            return {"key": 42}

        result = run_coroutine_thread_safe(returns_dict())
        assert isinstance(result, dict)
        assert result["key"] == 42

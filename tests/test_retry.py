"""Tests for retry decorator module."""

import asyncio
from unittest.mock import Mock

import pytest
from openai import APIError, RateLimitError

from openai_sdk_helpers.retry import with_exponential_backoff


def _create_rate_limit_error() -> RateLimitError:
    """Create a properly initialized RateLimitError for testing."""
    response = Mock()
    response.status_code = 429
    return RateLimitError("Rate limited", response=response, body={})


class TestRetryDecorator:
    """Test exponential backoff retry decorator."""

    def test_sync_function_succeeds_first_try(self) -> None:
        """Should return immediately if function succeeds."""

        @with_exponential_backoff(max_retries=3)
        def successful_func() -> str:
            return "success"

        result = successful_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_function_succeeds_first_try(self) -> None:
        """Should return immediately if async function succeeds."""

        @with_exponential_backoff(max_retries=3)
        async def successful_async() -> str:
            return "success"

        result = await successful_async()
        assert result == "success"

    def test_sync_function_retries_on_rate_limit(self) -> None:
        """Should retry on RateLimitError."""
        call_count = 0

        @with_exponential_backoff(max_retries=2, base_delay=0.01)
        def sometimes_fails() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise _create_rate_limit_error()
            return "success"

        result = sometimes_fails()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_function_retries_on_rate_limit(self) -> None:
        """Should retry async function on RateLimitError."""
        call_count = 0

        @with_exponential_backoff(max_retries=2, base_delay=0.01)
        async def sometimes_fails_async() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise _create_rate_limit_error()
            await asyncio.sleep(0.001)
            return "success"

        result = await sometimes_fails_async()
        assert result == "success"
        assert call_count == 2

    def test_sync_function_does_not_retry_on_permanent_error(self) -> None:
        """Should not retry on permanent errors (non-API)."""

        @with_exponential_backoff(max_retries=3, base_delay=0.01)
        def raises_value_error() -> str:
            raise ValueError("Permanent error")

        with pytest.raises(ValueError):
            raises_value_error()

    def test_sync_function_fails_after_max_retries(self) -> None:
        """Should give up after max_retries."""
        call_count = 0

        @with_exponential_backoff(max_retries=2, base_delay=0.01)
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise _create_rate_limit_error()

        with pytest.raises(RateLimitError):
            always_fails()
        # max_retries=2 means 3 total attempts (initial + 2 retries)
        assert call_count == 3

    def test_sync_function_with_parameters(self) -> None:
        """Should pass parameters through correctly."""
        call_count = 0

        @with_exponential_backoff(max_retries=1, base_delay=0.01)
        def func_with_args(a: int, b: str, c: bool = False) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _create_rate_limit_error()
            return f"{a}-{b}-{c}"

        result = func_with_args(42, "test", c=True)
        assert result == "42-test-True"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_function_with_parameters(self) -> None:
        """Should pass parameters through correctly for async."""
        call_count = 0

        @with_exponential_backoff(max_retries=1, base_delay=0.01)
        async def async_func_with_args(x: int, y: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _create_rate_limit_error()
            return f"{x}-{y}"

        result = await async_func_with_args(100, "data")
        assert result == "100-data"
        assert call_count == 2

    def test_max_delay_is_respected(self) -> None:
        """Should not exceed max_delay between retries."""
        call_count = 0

        @with_exponential_backoff(max_retries=3, base_delay=100, max_delay=0.05)
        def func_with_delay() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _create_rate_limit_error()
            return "success"

        # Should complete quickly because max_delay caps the sleep
        result = func_with_delay()
        assert result == "success"

    def test_returns_correct_type(self) -> None:
        """Should preserve return type through decorator."""

        @with_exponential_backoff(max_retries=1)
        def returns_dict() -> dict[str, int]:
            return {"key": 42}

        result = returns_dict()
        assert isinstance(result, dict)
        assert result["key"] == 42

    @pytest.mark.asyncio
    async def test_async_returns_correct_type(self) -> None:
        """Should preserve return type for async functions."""

        @with_exponential_backoff(max_retries=1)
        async def async_returns_list() -> list[str]:
            return ["a", "b", "c"]

        result = await async_returns_list()
        assert isinstance(result, list)
        assert result == ["a", "b", "c"]

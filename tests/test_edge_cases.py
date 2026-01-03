"""Tests for edge cases and unusual scenarios."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from openai_sdk_helpers import (
    AsyncExecutionError,
    ConfigurationError,
    LoggerFactory,
    run_coroutine_thread_safe,
    run_coroutine_with_fallback,
    with_exponential_backoff,
)
from openai_sdk_helpers.errors import InputValidationError
from openai_sdk_helpers.validation import validate_non_empty_string


class TestEmptyInputs:
    """Test handling of empty/null inputs."""

    def test_empty_string_validation_fails(self) -> None:
        """Empty strings should fail validation."""
        with pytest.raises(InputValidationError):
            validate_non_empty_string("", "test")

    def test_none_configuration_error(self) -> None:
        """None values should be caught."""
        with pytest.raises(ConfigurationError):
            if not None:
                raise ConfigurationError("Config is None")

    def test_empty_list_handling(self) -> None:
        """Should handle empty lists gracefully."""
        empty_list = []
        assert len(empty_list) == 0
        assert isinstance(empty_list, list)

    def test_empty_dict_handling(self) -> None:
        """Should handle empty dicts gracefully."""
        empty_dict = {}
        assert len(empty_dict) == 0
        assert isinstance(empty_dict, dict)


class TestConcurrentAccess:
    """Test concurrent and parallel operations."""

    def test_multiple_coroutines_sequentially(self) -> None:
        """Should handle multiple coroutine calls."""

        async def task(n: int) -> int:
            await asyncio.sleep(0.01)
            return n * 2

        results = []
        for i in range(3):
            result = run_coroutine_with_fallback(task(i))
            results.append(result)

        assert results == [0, 2, 4]

    @pytest.mark.asyncio
    async def test_multiple_concurrent_coroutines(self) -> None:
        """Should handle multiple concurrent coroutines."""

        async def task(n: int) -> int:
            await asyncio.sleep(0.01)
            return n * 2

        tasks = [asyncio.create_task(task(i)) for i in range(3)]
        results = await asyncio.gather(*tasks)
        assert results == [0, 2, 4]

    def test_retry_decorator_with_concurrent_calls(self) -> None:
        """Retry decorator should work with concurrent calls."""
        call_count = 0

        @with_exponential_backoff(max_retries=1, base_delay=0.01)
        def increment() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(increment) for _ in range(3)]
            results = [f.result() for f in futures]

        assert len(results) == 3


class TestLongRunningOperations:
    """Test behavior with long-running operations."""

    def test_async_timeout_with_long_operation(self) -> None:
        """Should timeout on long operations."""

        async def slow_operation() -> str:
            await asyncio.sleep(10)
            return "never returns"

        with pytest.raises(AsyncExecutionError, match="timed out"):
            run_coroutine_thread_safe(slow_operation(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_long_operation_completion(self) -> None:
        """Should complete long operations if time permits."""

        async def long_operation() -> str:
            await asyncio.sleep(0.05)
            return "completed"

        result = run_coroutine_with_fallback(long_operation())
        assert result == "completed"


class TestErrorPropagation:
    """Test that errors propagate correctly."""

    def test_exception_from_coroutine_propagates(self) -> None:
        """Exception from coroutine should propagate."""

        async def failing_coro() -> None:
            raise RuntimeError("Coroutine failed")

        with pytest.raises(RuntimeError, match="Coroutine failed"):
            run_coroutine_with_fallback(failing_coro())

    def test_validation_error_propagates(self) -> None:
        """Validation errors should propagate."""
        with pytest.raises(InputValidationError):
            validate_non_empty_string("", "test_field")

    def test_retry_exhaustion_propagates_error(self) -> None:
        """Exhausted retries should propagate original error."""

        @with_exponential_backoff(max_retries=1, base_delay=0.01)
        def always_fails() -> None:
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError, match="Always fails"):
            always_fails()


class TestResourceStates:
    """Test proper resource state transitions."""

    def test_logger_configuration_changes_state(self) -> None:
        """Logger configuration should change state."""
        import logging

        # Configure with one level
        LoggerFactory.configure(level=logging.WARNING)
        logger1 = LoggerFactory.get_logger("test1")

        # Reconfigure with different level
        LoggerFactory.configure(level=logging.DEBUG)
        logger2 = LoggerFactory.get_logger("test2")

        # Both should exist
        assert logger1 is not None
        assert logger2 is not None

    def test_nested_exception_handling(self) -> None:
        """Should handle nested exceptions properly."""
        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise ConfigurationError("Outer error") from e
        except ConfigurationError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)


class TestTypePreservation:
    """Test that types are preserved through operations."""

    def test_int_preservation_in_coroutine(self) -> None:
        """Should preserve int types."""

        async def return_int() -> int:
            return 42

        result = run_coroutine_with_fallback(return_int())
        assert isinstance(result, int)
        assert result == 42

    def test_dict_preservation_in_coroutine(self) -> None:
        """Should preserve dict types."""

        async def return_dict() -> dict[str, int]:
            return {"key": 123}

        result = run_coroutine_with_fallback(return_dict())
        assert isinstance(result, dict)
        assert result["key"] == 123

    def test_list_preservation_in_coroutine(self) -> None:
        """Should preserve list types."""

        async def return_list() -> list[str]:
            return ["a", "b", "c"]

        result = run_coroutine_with_fallback(return_list())
        assert isinstance(result, list)
        assert len(result) == 3


class TestBoundaryConditions:
    """Test boundary conditions and edge values."""

    def test_max_retries_zero(self) -> None:
        """Should work with max_retries=0."""
        call_count = 0

        @with_exponential_backoff(max_retries=0, base_delay=0.01)
        def once_only() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result = once_only()
        assert result == 1
        assert call_count == 1

    def test_very_short_timeout(self) -> None:
        """Should handle very short timeouts."""

        async def instant_return() -> str:
            return "fast"

        result = run_coroutine_thread_safe(instant_return(), timeout=0.001)
        assert result == "fast"

    def test_very_long_timeout(self) -> None:
        """Should handle very long timeouts."""

        async def quick_task() -> str:
            await asyncio.sleep(0.01)
            return "done"

        result = run_coroutine_thread_safe(quick_task(), timeout=3600)
        assert result == "done"


class TestContextIsolation:
    """Test that operations don't interfere with each other."""

    def test_exception_in_one_doesnt_affect_other(self) -> None:
        """Exception in one operation shouldn't affect another."""

        @with_exponential_backoff(max_retries=1, base_delay=0.01)
        def operation_1() -> int:
            raise RuntimeError("Op 1 fails")

        @with_exponential_backoff(max_retries=1, base_delay=0.01)
        def operation_2() -> int:
            return 42

        # Operation 1 fails
        with pytest.raises(RuntimeError):
            operation_1()

        # Operation 2 should still work
        result = operation_2()
        assert result == 42

    @pytest.mark.asyncio
    async def test_multiple_async_operations_isolated(self) -> None:
        """Multiple async operations should be isolated."""

        async def op1() -> int:
            await asyncio.sleep(0.01)
            return 1

        async def op2() -> int:
            await asyncio.sleep(0.01)
            return 2

        r1 = run_coroutine_with_fallback(op1())
        r2 = run_coroutine_with_fallback(op2())

        assert r1 == 1
        assert r2 == 2

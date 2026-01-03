"""Tests for context manager utilities module."""

import pytest

from openai_sdk_helpers.context_manager import (
    AsyncManagedResource,
    ManagedResource,
    ensure_closed,
    ensure_closed_async,
)


class TestResource(ManagedResource):
    """Test implementation of ManagedResource."""

    def __init__(self):
        self.closed = False
        self.close_called = 0

    def close(self) -> None:
        self.closed = True
        self.close_called += 1


class TestAsyncResource(AsyncManagedResource):
    """Test implementation of AsyncManagedResource."""

    def __init__(self):
        self.closed = False
        self.close_called = 0

    async def close(self) -> None:
        self.closed = True
        self.close_called += 1


class TestManagedResource:
    """Test synchronous resource management."""

    def test_context_manager_calls_close(self) -> None:
        """Should call close() on exit."""
        resource = TestResource()
        with resource:
            assert not resource.closed
        assert resource.closed

    def test_context_manager_calls_close_on_exception(self) -> None:
        """Should call close() even if exception occurs."""
        resource = TestResource()
        try:
            with resource:
                raise ValueError("Test error")
        except ValueError:
            pass
        assert resource.closed

    def test_context_manager_returns_self(self) -> None:
        """Should return self when entering context."""
        resource = TestResource()
        with resource as r:
            assert r is resource

    def test_context_manager_re_raises_exceptions(self) -> None:
        """Should re-raise exceptions from with block."""
        resource = TestResource()
        with pytest.raises(RuntimeError):
            with resource:
                raise RuntimeError("Test error")

    def test_multiple_closes_safe(self) -> None:
        """Should be safe to call close multiple times."""
        resource = TestResource()
        with resource:
            pass
        # Can call close again without error
        resource.close()
        assert resource.close_called == 2


class TestAsyncManagedResource:
    """Test asynchronous resource management."""

    @pytest.mark.asyncio
    async def test_async_context_manager_calls_close(self) -> None:
        """Should call close() on exit."""
        resource = TestAsyncResource()
        async with resource:
            assert not resource.closed
        assert resource.closed

    @pytest.mark.asyncio
    async def test_async_context_manager_calls_close_on_exception(self) -> None:
        """Should call close() even if exception occurs."""
        resource = TestAsyncResource()
        try:
            async with resource:
                raise ValueError("Test error")
        except ValueError:
            pass
        assert resource.closed

    @pytest.mark.asyncio
    async def test_async_context_manager_returns_self(self) -> None:
        """Should return self when entering async context."""
        resource = TestAsyncResource()
        async with resource as r:
            assert r is resource

    @pytest.mark.asyncio
    async def test_async_context_manager_re_raises_exceptions(self) -> None:
        """Should re-raise exceptions from async with block."""
        resource = TestAsyncResource()
        with pytest.raises(RuntimeError):
            async with resource:
                raise RuntimeError("Test error")


class TestEnsureClosed:
    """Test ensure_closed helper."""

    def test_closes_resource_with_close_method(self) -> None:
        """Should close resource that has close method."""

        class CloseableResource:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        resource = CloseableResource()
        ensure_closed(resource)
        assert resource.closed

    def test_handles_none_gracefully(self) -> None:
        """Should handle None without error."""
        ensure_closed(None)  # Should not raise

    def test_handles_object_without_close_method(self) -> None:
        """Should handle objects without close method."""

        class NoCloseResource:
            pass

        resource = NoCloseResource()
        ensure_closed(resource)  # Should not raise

    def test_logs_error_but_doesnt_raise(self) -> None:
        """Should log errors but not raise them."""

        class FailingResource:
            def close(self):
                raise RuntimeError("Close failed")

        resource = FailingResource()
        ensure_closed(resource)  # Should not raise


class TestEnsureClosedAsync:
    """Test ensure_closed_async helper."""

    @pytest.mark.asyncio
    async def test_closes_async_resource(self) -> None:
        """Should close resource with async close method."""

        class AsyncCloseableResource:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        resource = AsyncCloseableResource()
        await ensure_closed_async(resource)
        assert resource.closed

    @pytest.mark.asyncio
    async def test_closes_sync_resource(self) -> None:
        """Should close resource with sync close method."""

        class SyncCloseableResource:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        resource = SyncCloseableResource()
        await ensure_closed_async(resource)
        assert resource.closed

    @pytest.mark.asyncio
    async def test_handles_none_gracefully(self) -> None:
        """Should handle None without error."""
        await ensure_closed_async(None)  # Should not raise

    @pytest.mark.asyncio
    async def test_logs_error_but_doesnt_raise(self) -> None:
        """Should log errors but not raise them."""

        class FailingResource:
            async def close(self):
                raise RuntimeError("Close failed")

        resource = FailingResource()
        await ensure_closed_async(resource)  # Should not raise

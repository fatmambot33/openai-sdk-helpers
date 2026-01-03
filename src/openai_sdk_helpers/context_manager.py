"""Context manager utilities for resource cleanup.

Provides base classes and utilities for proper resource management
with guaranteed cleanup on exit or exception.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, AsyncIterator, Generic, Optional, TypeVar

from openai_sdk_helpers.utils.core import log

T = TypeVar("T")


class ManagedResource(Generic[T]):
    """Base class for resources that need cleanup.

    Provides context manager support for guaranteed resource cleanup
    even when exceptions occur.

    Examples
    --------
    >>> class DatabaseConnection(ManagedResource[Connection]):
    ...     def __init__(self, connection):
    ...         self.connection = connection
    ...
    ...     def close(self) -> None:
    ...         if self.connection:
    ...             self.connection.close()

    >>> with DatabaseConnection(connect()) as db:
    ...     db.query("SELECT ...")
    """

    def __enter__(self) -> T:
        """Enter context manager.

        Returns
        -------
        T
            The resource instance (self cast appropriately).
        """
        return self  # type: ignore

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Exit context manager with cleanup.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Type of exception if one was raised, None otherwise.
        exc_val : BaseException | None
            Exception instance if one was raised, None otherwise.
        exc_tb : TracebackType | None
            Traceback if exception was raised, None otherwise.

        Returns
        -------
        bool
            False to re-raise exceptions, True to suppress them.
        """
        try:
            self.close()
        except Exception as exc:
            log(f"Error during cleanup: {exc}", level=30)  # logging.WARNING
            # Don't suppress cleanup errors
            if exc_type is None:
                raise

        return False  # Re-raise exceptions

    def close(self) -> None:
        """Close and cleanup the resource.

        Should be overridden by subclasses to perform actual cleanup.
        Should not raise exceptions, but may log them.

        Raises
        ------
        Exception
            May raise if cleanup fails catastrophically.
        """
        pass


class AsyncManagedResource(Generic[T]):
    """Base class for async resources that need cleanup.

    Provides async context manager support for guaranteed resource cleanup
    even when exceptions occur.

    Examples
    --------
    >>> class AsyncDatabaseConnection(AsyncManagedResource[AsyncConnection]):
    ...     def __init__(self, connection):
    ...         self.connection = connection
    ...
    ...     async def close(self) -> None:
    ...         if self.connection:
    ...             await self.connection.close()

    >>> async with AsyncDatabaseConnection(await connect()) as db:
    ...     await db.query("SELECT ...")
    """

    async def __aenter__(self) -> T:
        """Enter async context manager.

        Returns
        -------
        T
            The resource instance (self cast appropriately).
        """
        return self  # type: ignore

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Exit async context manager with cleanup.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Type of exception if one was raised, None otherwise.
        exc_val : BaseException | None
            Exception instance if one was raised, None otherwise.
        exc_tb : TracebackType | None
            Traceback if exception was raised, None otherwise.

        Returns
        -------
        bool
            False to re-raise exceptions, True to suppress them.
        """
        try:
            await self.close()
        except Exception as exc:
            log(f"Error during async cleanup: {exc}", level=30)  # logging.WARNING
            # Don't suppress cleanup errors
            if exc_type is None:
                raise

        return False  # Re-raise exceptions

    async def close(self) -> None:
        """Close and cleanup the resource asynchronously.

        Should be overridden by subclasses to perform actual cleanup.
        Should not raise exceptions, but may log them.

        Raises
        ------
        Exception
            May raise if cleanup fails catastrophically.
        """
        pass


def ensure_closed(resource: Any) -> None:
    """Safely close a resource if it has a close method.

    Logs errors but doesn't raise them.

    Parameters
    ----------
    resource : Any
        Object that may have a close() method.
    """
    if resource is None:
        return

    close_method = getattr(resource, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception as exc:
            log(f"Error closing {type(resource).__name__}: {exc}", level=30)


async def ensure_closed_async(resource: Any) -> None:
    """Safely close a resource asynchronously if it has an async close method.

    Logs errors but doesn't raise them.

    Parameters
    ----------
    resource : Any
        Object that may have an async close() method.
    """
    if resource is None:
        return

    close_method = getattr(resource, "close", None)
    if callable(close_method):
        try:
            if asyncio.iscoroutinefunction(close_method):
                await close_method()
            else:
                close_method()
        except Exception as exc:
            log(
                f"Error closing async {type(resource).__name__}: {exc}",
                level=30,
            )


@asynccontextmanager
async def async_context(resource: AsyncManagedResource[T]) -> AsyncIterator[T]:
    """Context manager for async resources.

    Parameters
    ----------
    resource : AsyncManagedResource
        Async resource to manage.

    Yields
    ------
    T
        The resource instance.

    Examples
    --------
    >>> async with async_context(my_resource) as resource:
    ...     await resource.do_something()
    """
    try:
        yield await resource.__aenter__()
    finally:
        await resource.__aexit__(None, None, None)

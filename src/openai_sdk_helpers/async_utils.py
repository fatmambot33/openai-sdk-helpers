"""Async/sync bridge utilities with proper error handling.

Provides thread-safe wrappers for running async code from sync contexts
with proper exception propagation and timeout handling.
"""

import asyncio
import queue
import threading
from typing import Any, Coroutine, Generic, TypeVar

from openai_sdk_helpers.errors import AsyncExecutionError
from openai_sdk_helpers.utils.core import log

T = TypeVar("T")

# Default timeout constants
DEFAULT_COROUTINE_TIMEOUT = 300.0  # 5 minutes
THREAD_JOIN_TIMEOUT = 5.0  # 5 seconds


def run_coroutine_thread_safe(
    coro: Coroutine[Any, Any, T],
    timeout: float = DEFAULT_COROUTINE_TIMEOUT,
) -> T:
    """Run a coroutine in a thread-safe manner from a sync context.

    Uses a queue to safely communicate results and exceptions between threads.
    Ensures exceptions from the async operation are properly propagated.

    Parameters
    ----------
    coro : Coroutine
        The coroutine to execute.
    timeout : float
        Maximum time in seconds to wait for the coroutine to complete.
        Default is 300 (5 minutes).

    Returns
    -------
    Any
        Result from the coroutine.

    Raises
    ------
    AsyncExecutionError
        If the coroutine fails or timeout occurs.

    Examples
    --------
    >>> async def fetch_data():
    ...     return "data"
    >>> result = run_coroutine_thread_safe(fetch_data())
    """
    result_queue: queue.Queue[T | Exception] = queue.Queue()

    def _thread_runner() -> None:
        """Run coroutine and put result in queue."""
        try:
            result = asyncio.run(coro)
            result_queue.put(result)
        except Exception as exc:
            # Queue stores the exception to propagate later
            result_queue.put(exc)

    thread = threading.Thread(target=_thread_runner, daemon=False)
    thread.start()

    try:
        result = result_queue.get(timeout=timeout)
        if isinstance(result, Exception):
            # Re-raise the exception from the thread
            raise result
        return result
    except queue.Empty:
        raise AsyncExecutionError(
            f"Coroutine execution timed out after {timeout} seconds"
        ) from None
    finally:
        # Ensure thread is cleaned up
        thread.join(timeout=THREAD_JOIN_TIMEOUT)
        if thread.is_alive():
            log(
                f"Thread {thread.name} did not terminate within {THREAD_JOIN_TIMEOUT} seconds",
                level=20,  # logging.INFO
            )


def run_coroutine_with_fallback(
    coro: Coroutine[Any, Any, T],
) -> T:
    """Run a coroutine, falling back to thread if event loop is already running.

    Attempts to run the coroutine directly if no event loop is present.
    If an event loop is already running (nested scenario), creates a new
    thread to avoid the "RuntimeError: asyncio.run() cannot be called from a
    running event loop" error.

    Parameters
    ----------
    coro : Coroutine
        The coroutine to execute.

    Returns
    -------
    Any
        Result from the coroutine.

    Raises
    ------
    AsyncExecutionError
        If execution fails or times out.

    Examples
    --------
    >>> async def fetch_data():
    ...     return "data"
    >>> result = run_coroutine_with_fallback(fetch_data())
    """
    try:
        # Try to get currently running loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)

    # Loop is already running, must use thread
    if loop.is_running():
        return run_coroutine_thread_safe(coro)

    # This shouldn't happen but handle defensive
    return loop.run_until_complete(coro)

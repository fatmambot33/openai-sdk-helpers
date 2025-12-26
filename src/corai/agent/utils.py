"""Utility helpers for synchronous interaction with async agents."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_coro_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from synchronous code.

    Parameters
    ----------
    coro : Coroutine[Any, Any, T]
        Coroutine to execute.

    Returns
    -------
    T
        Result returned by the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop.is_running():
        result: T | None = None

        def _thread_runner() -> None:
            nonlocal result
            result = asyncio.run(coro)

        thread = threading.Thread(target=_thread_runner, daemon=True)
        thread.start()
        thread.join()
        if result is None:
            raise RuntimeError("Coroutine execution did not return a result.")
        return result

    return loop.run_until_complete(coro)


__all__ = ["run_coro_sync"]

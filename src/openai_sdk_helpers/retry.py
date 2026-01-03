"""Retry decorators with exponential backoff for API operations.

Provides decorators for retrying async and sync functions with
exponential backoff and jitter when rate limiting or transient
errors occur.
"""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from openai import APIError, RateLimitError

from openai_sdk_helpers.errors import AsyncExecutionError
from openai_sdk_helpers.utils.core import log

P = ParamSpec("P")
T = TypeVar("T")

# Default retry configuration constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0

# HTTP status codes for transient errors
TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503})


def with_exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate functions with exponential backoff on transient errors.

    Retries on RateLimitError or transient API errors (5xx, 408, 429).
    Uses exponential backoff with jitter to avoid thundering herd.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (total attempts = max_retries + 1).
        Default is 3.
    base_delay : float
        Initial delay in seconds before first retry. Default is 1.0.
    max_delay : float
        Maximum delay in seconds between retries. Default is 60.0.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @with_exponential_backoff(max_retries=3, base_delay=1.0)
    ... def call_api(query: str) -> str:
    ...     # API call that may fail with rate limiting
    ...     return client.call(query)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        """Apply retry logic to function."""
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                """Async wrapper with retry logic."""
                last_exc: Exception | None = None
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except RateLimitError as exc:
                        last_exc = exc
                        if attempt >= max_retries:
                            raise
                        delay = min(
                            base_delay * (2**attempt) + random.uniform(0, 1),
                            max_delay,
                        )
                        log(
                            f"Rate limited on {func.__name__}, retrying in "
                            f"{delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})",
                            level=logging.WARNING,
                        )
                        await asyncio.sleep(delay)
                    except APIError as exc:
                        last_exc = exc
                        status_code: int | None = getattr(exc, "status_code", None)
                        # Only retry on transient errors
                        if (
                            not status_code
                            or status_code not in TRANSIENT_HTTP_STATUS_CODES
                        ):
                            raise
                        if attempt >= max_retries:
                            raise
                        delay = min(
                            base_delay * (2**attempt),
                            max_delay,
                        )
                        log(
                            f"Transient API error on {func.__name__}: "
                            f"{status_code}, retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{max_retries + 1})",
                            level=logging.WARNING,
                        )
                        await asyncio.sleep(delay)

                # Should never reach here, but handle edge case
                if last_exc:
                    raise last_exc
                raise AsyncExecutionError(
                    f"Unexpected state in {func.__name__} after retries"
                )

            return async_wrapper  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """Sync wrapper with retry logic."""
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        raise
                    delay = min(
                        base_delay * (2**attempt) + random.uniform(0, 1),
                        max_delay,
                    )
                    log(
                        f"Rate limited on {func.__name__}, retrying in "
                        f"{delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})",
                        level=logging.WARNING,
                    )
                    time.sleep(delay)
                except APIError as exc:
                    last_exc = exc
                    status_code: int | None = getattr(exc, "status_code", None)
                    # Only retry on transient errors
                    if (
                        not status_code
                        or status_code not in TRANSIENT_HTTP_STATUS_CODES
                    ):
                        raise
                    if attempt >= max_retries:
                        raise
                    delay = min(
                        base_delay * (2**attempt),
                        max_delay,
                    )
                    log(
                        f"Transient API error on {func.__name__}: "
                        f"{status_code}, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{max_retries + 1})",
                        level=logging.WARNING,
                    )
                    time.sleep(delay)

            # Should never reach here, but handle edge case
            if last_exc:
                raise last_exc
            raise AsyncExecutionError(
                f"Unexpected state in {func.__name__} after retries"
            )

        return sync_wrapper  # type: ignore

    return decorator

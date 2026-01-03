"""Convenience functions for executing response workflows.

This module provides high-level functions that handle the complete lifecycle
of response workflows including instantiation, execution, and resource cleanup.
They simplify common usage patterns for both synchronous and asynchronous contexts.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from .base import BaseResponse


R = TypeVar("R", bound=BaseResponse[Any])


def run_sync(
    response_cls: type[R],
    *,
    content: str,
    response_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Execute a response workflow synchronously with automatic cleanup.

    Instantiates the response class, executes run_sync with the provided
    content, and ensures cleanup occurs even if an exception is raised.

    Parameters
    ----------
    response_cls : type[BaseResponse]
        Response class to instantiate for the workflow.
    content : str
        Prompt text to send to the OpenAI API.
    response_kwargs : dict[str, Any] or None, default None
        Optional keyword arguments forwarded to response_cls constructor.

    Returns
    -------
    Any
        Parsed response from BaseResponse.run_sync, typically a structured
        output or None.

    Examples
    --------
    >>> from openai_sdk_helpers.response import run_sync
    >>> result = run_sync(
    ...     MyResponse,
    ...     content="Analyze this text",
    ...     response_kwargs={"openai_settings": settings}
    ... )
    """
    response = response_cls(**(response_kwargs or {}))
    try:
        return response.run_sync(content=content)
    finally:
        response.close()


async def run_async(
    response_cls: type[R],
    *,
    content: str,
    response_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Execute a response workflow asynchronously with automatic cleanup.

    Instantiates the response class, executes run_async with the provided
    content, and ensures cleanup occurs even if an exception is raised.

    Parameters
    ----------
    response_cls : type[BaseResponse]
        Response class to instantiate for the workflow.
    content : str
        Prompt text to send to the OpenAI API.
    response_kwargs : dict[str, Any] or None, default None
        Optional keyword arguments forwarded to response_cls constructor.

    Returns
    -------
    Any
        Parsed response from BaseResponse.run_async, typically a structured
        output or None.

    Examples
    --------
    >>> from openai_sdk_helpers.response import run_async
    >>> result = await run_async(
    ...     MyResponse,
    ...     content="Summarize this document",
    ...     response_kwargs={"openai_settings": settings}
    ... )
    """
    response = response_cls(**(response_kwargs or {}))
    try:
        return await response.run_async(content=content)
    finally:
        response.close()


def run_streamed(
    response_cls: type[R],
    *,
    content: str,
    response_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Execute a response workflow and return the awaited result.

    Provides API compatibility with agent interfaces. Streaming responses
    are not currently fully supported, so this executes run_async and
    awaits the result.

    Parameters
    ----------
    response_cls : type[BaseResponse]
        Response class to instantiate for the workflow.
    content : str
        Prompt text to send to the OpenAI API.
    response_kwargs : dict[str, Any] or None, default None
        Optional keyword arguments forwarded to response_cls constructor.

    Returns
    -------
    Any
        Parsed response from run_async, typically a structured output or None.

    Notes
    -----
    This function exists for API consistency but does not currently provide
    true streaming functionality.

    Examples
    --------
    >>> from openai_sdk_helpers.response import run_streamed
    >>> result = run_streamed(
    ...     MyResponse,
    ...     content="Process this text",
    ...     response_kwargs={"openai_settings": settings}
    ... )
    """
    return asyncio.run(
        run_async(response_cls, content=content, response_kwargs=response_kwargs)
    )


__all__ = ["run_sync", "run_async", "run_streamed"]

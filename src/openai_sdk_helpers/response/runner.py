"""Convenience functions for executing response workflows.

This module provides high-level functions that handle the complete lifecycle
of response workflows including instantiation, execution, and resource cleanup.
They simplify common usage patterns for both synchronous and asynchronous contexts.
"""

from __future__ import annotations

from typing import Any, TypeVar

from .base import ResponseBase

R = TypeVar("R", bound=ResponseBase[Any])


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
    response_cls : type[ResponseBase]
        Response class to instantiate for the workflow.
    content : str
        Prompt text to send to the OpenAI API.
    response_kwargs : dict[str, Any] or None, default None
        Optional keyword arguments forwarded to response_cls constructor.

    Returns
    -------
    Any
        Parsed response from ResponseBase.run_sync, typically a structured
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
    response_cls : type[ResponseBase]
        Response class to instantiate for the workflow.
    content : str
        Prompt text to send to the OpenAI API.
    response_kwargs : dict[str, Any] or None, default None
        Optional keyword arguments forwarded to response_cls constructor.

    Returns
    -------
    Any
        Parsed response from ResponseBase.run_async, typically a structured
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


__all__ = ["run_sync", "run_async"]

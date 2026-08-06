"""Convenience functions for executing response workflows.

This module provides high-level functions that handle the complete lifecycle
of response workflows including instantiation, execution, and resource cleanup.
They simplify common usage patterns for both synchronous and asynchronous contexts.
"""

from __future__ import annotations

from typing import Any, TypeVar

from openai_sdk_helpers.runtime import (
    OperationContext,
    run_observed_async,
    run_observed_sync,
)

from .base import ResponseBase

R = TypeVar("R", bound=ResponseBase[Any])


def run_sync(
    response_cls: type[R],
    *,
    content: str,
    response_kwargs: dict[str, Any] | None = None,
    operation_context: OperationContext | None = None,
) -> Any:
    """Execute a response workflow synchronously with automatic cleanup.

    Instantiates the response class, executes ``run_sync`` with the provided
    content, and ensures cleanup occurs even if an exception is raised.

    Parameters
    ----------
    response_cls : type[ResponseBase]
        Response class to instantiate for the workflow.
    content : str
        Prompt text to send to the OpenAI API.
    response_kwargs : dict[str, Any] or None, default=None
        Optional keyword arguments forwarded to the response constructor.
    operation_context : OperationContext or None, default=None
        Optional explicit lifecycle and observability context.

    Returns
    -------
    Any
        Original parsed result from ``ResponseBase.run_sync``.

    Examples
    --------
    >>> from openai_sdk_helpers.response import run_sync
    >>> result = run_sync(
    ...     MyResponse,
    ...     content="Analyze this text",
    ...     response_kwargs={"openai_settings": settings},
    ... )
    """

    def execute() -> Any:
        response = response_cls(**(response_kwargs or {}))
        try:
            return response.run_sync(content=content)
        finally:
            response.close()

    return run_observed_sync(operation_context, execute)


async def run_async(
    response_cls: type[R],
    *,
    content: str,
    response_kwargs: dict[str, Any] | None = None,
    operation_context: OperationContext | None = None,
) -> Any:
    """Execute a response workflow asynchronously with automatic cleanup.

    Instantiates the response class, executes ``run_async`` with the provided
    content, and ensures cleanup occurs even if an exception is raised.

    Parameters
    ----------
    response_cls : type[ResponseBase]
        Response class to instantiate for the workflow.
    content : str
        Prompt text to send to the OpenAI API.
    response_kwargs : dict[str, Any] or None, default=None
        Optional keyword arguments forwarded to the response constructor.
    operation_context : OperationContext or None, default=None
        Optional explicit lifecycle and observability context.

    Returns
    -------
    Any
        Original parsed result from ``ResponseBase.run_async``.

    Examples
    --------
    >>> from openai_sdk_helpers.response import run_async
    >>> result = await run_async(
    ...     MyResponse,
    ...     content="Summarize this document",
    ...     response_kwargs={"openai_settings": settings},
    ... )
    """

    async def execute() -> Any:
        response = response_cls(**(response_kwargs or {}))
        try:
            return await response.run_async(content=content)
        finally:
            response.close()

    return await run_observed_async(operation_context, execute)


__all__ = ["run_sync", "run_async"]

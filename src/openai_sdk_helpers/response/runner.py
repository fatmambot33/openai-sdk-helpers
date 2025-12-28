"""Convenience runners for response workflows."""

from __future__ import annotations

import asyncio

from typing import Any, Optional, Type, TypeVar

from .base import ResponseBase

T = TypeVar("T")
R = TypeVar("R", bound=ResponseBase[Any])


def run(
    response_cls: Type[R],
    *,
    content: str,
    response_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """Run a response workflow synchronously and close resources.

    Parameters
    ----------
    response_cls
        Response class to instantiate.
    content
        Prompt text to send to the OpenAI API.
    response_kwargs
        Keyword arguments forwarded to ``response_cls``. Default ``None``.

    Returns
    -------
    Any
        Parsed response from :meth:`ResponseBase.run_response`.
    """
    response = response_cls(**(response_kwargs or {}))
    try:
        return response.run_response(content=content)
    finally:
        response.close()


async def run_async(
    response_cls: Type[R],
    *,
    content: str,
    response_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """Run a response workflow asynchronously and close resources.

    Parameters
    ----------
    response_cls
        Response class to instantiate.
    content
        Prompt text to send to the OpenAI API.
    response_kwargs
        Keyword arguments forwarded to ``response_cls``. Default ``None``.

    Returns
    -------
    Any
        Parsed response from :meth:`ResponseBase.run_response_async`.
    """
    response = response_cls(**(response_kwargs or {}))
    try:
        return await response.run_response_async(content=content)
    finally:
        response.close()


def run_streamed(
    response_cls: Type[R],
    *,
    content: str,
    response_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """Run a response workflow and return the asynchronous result.

    This mirrors the agent API for discoverability. Streaming responses are not
    currently supported by :class:`ResponseBase`, so this returns the same value
    as :func:`run_async`.

    Parameters
    ----------
    response_cls
        Response class to instantiate.
    content
        Prompt text to send to the OpenAI API.
    response_kwargs
        Keyword arguments forwarded to ``response_cls``. Default ``None``.

    Returns
    -------
    Any
        Parsed response returned from :func:`run_async`.
    """
    return asyncio.run(
        run_async(response_cls, content=content, response_kwargs=response_kwargs)
    )


__all__ = ["run", "run_async", "run_streamed"]

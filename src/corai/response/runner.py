"""Convenience runners for response workflows."""

from __future__ import annotations

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
    response_cls : type[ResponseBase]
        Response class to instantiate.
    content : str
        Prompt text to send to the OpenAI API.
    response_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to ``response_cls``.

    Returns
    -------
    Any
        Parsed response from :meth:`ResponseBase.generate_response`.
    """
    response = response_cls(**(response_kwargs or {}))
    try:
        return response.generate_response(content=content)
    finally:
        response.close()


__all__ = ["run"]

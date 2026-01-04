"""Tool handler utilities for OpenAI SDK interactions.

This module provides generic tool handling infrastructure including argument
parsing, Pydantic validation, function execution, and result serialization.
These utilities reduce boilerplate and ensure consistent tool behavior.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from openai_sdk_helpers.response.tool_call import parse_tool_arguments

T = TypeVar("T", bound=BaseModel)


def serialize_tool_result(result: Any) -> str:
    """Serialize tool results into a standardized JSON string.

    Handles Pydantic models, lists, dicts, and plain strings with consistent
    JSON formatting. Pydantic models are serialized using model_dump(),
    while other types are converted to JSON or string representation.

    Parameters
    ----------
    result : Any
        Tool result to serialize. Can be a Pydantic model, list, dict, str,
        or any JSON-serializable type.

    Returns
    -------
    str
        JSON-formatted string representation of the result.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class Result(BaseModel):
    ...     value: int
    >>> serialize_tool_result(Result(value=42))
    '{"value": 42}'

    >>> serialize_tool_result(["item1", "item2"])
    '["item1", "item2"]'

    >>> serialize_tool_result("plain text")
    '"plain text"'

    >>> serialize_tool_result({"key": "value"})
    '{"key": "value"}'
    """
    # Handle Pydantic models
    if isinstance(result, BaseModel):
        return result.model_dump_json()

    # Handle strings - wrap in JSON string format
    if isinstance(result, str):
        return json.dumps(result)

    # Handle other JSON-serializable types (lists, dicts, primitives)
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        # Fallback to string representation for non-JSON types
        return json.dumps(str(result))


def tool_handler_factory(
    func: Callable[..., Any],
    input_model: type[T] | None = None,
) -> Callable[[Any], str]:
    """Create a generic tool handler that parses, validates, and serializes.

    Wraps a tool function with automatic argument parsing, optional Pydantic
    validation, execution, and result serialization. This eliminates
    repetitive boilerplate for tool implementations.

    The returned handler:
    1. Parses tool_call.arguments using parse_tool_arguments
    2. Validates arguments with input_model if provided
    3. Calls func with validated/parsed arguments
    4. Serializes the result using serialize_tool_result

    Parameters
    ----------
    func : Callable[..., Any]
        The actual tool implementation function. Should accept keyword
        arguments matching the tool's parameter schema. Can be synchronous
        or asynchronous.
    input_model : type[BaseModel] or None, default None
        Optional Pydantic model for input validation. When provided,
        arguments are validated and converted to this model before being
        passed to func.

    Returns
    -------
    Callable[[Any], str]
        Handler function that accepts a tool_call object (with arguments
        and name attributes) and returns a JSON string result.

    Raises
    ------
    ValidationError
        If input_model is provided and validation fails.
    ValueError
        If argument parsing fails.

    Examples
    --------
    Basic usage without validation:

    >>> def search_tool(query: str, limit: int = 10):
    ...     return {"results": [f"Result for {query}"]}
    >>> handler = tool_handler_factory(search_tool)

    With Pydantic validation:

    >>> from pydantic import BaseModel
    >>> class SearchInput(BaseModel):
    ...     query: str
    ...     limit: int = 10
    >>> def search_tool(query: str, limit: int = 10):
    ...     return {"results": [f"Result for {query}"]}
    >>> handler = tool_handler_factory(search_tool, SearchInput)

    The handler can then be used with OpenAI tool calls:

    >>> class ToolCall:
    ...     def __init__(self):
    ...         self.arguments = '{"query": "test", "limit": 5}'
    ...         self.name = "search"
    >>> tool_call = ToolCall()
    >>> result = handler(tool_call)  # doctest: +SKIP
    """

    def handler(tool_call: Any) -> str:
        """Handle tool execution with parsing, validation, and serialization.

        Parameters
        ----------
        tool_call : Any
            Tool call object with 'arguments' and 'name' attributes.

        Returns
        -------
        str
            JSON-formatted result from the tool function.

        Raises
        ------
        ValueError
            If argument parsing fails.
        ValidationError
            If Pydantic validation fails (when input_model is provided).
        """
        # Extract tool name for error context
        tool_name = getattr(tool_call, "name", None)

        # Parse arguments with error context
        parsed_args = parse_tool_arguments(tool_call.arguments, tool_name=tool_name)

        # Validate with Pydantic if model provided
        if input_model is not None:
            try:
                validated_input = input_model(**parsed_args)
                # Convert back to dict for function call
                call_kwargs = validated_input.model_dump()
            except ValidationError as exc:
                # Re-raise the original ValidationError with added context in message
                raise exc
        else:
            call_kwargs = parsed_args

        # Execute function (handle async if needed)
        if inspect.iscoroutinefunction(func):
            # Note: For async functions, the handler itself should be awaited
            # by the caller. We can't await here in a sync context.
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, create a task
                result = loop.create_task(func(**call_kwargs))
            except RuntimeError:
                # No event loop running, use asyncio.run
                result = asyncio.run(func(**call_kwargs))
        else:
            result = func(**call_kwargs)

        # Serialize result
        return serialize_tool_result(result)

    return handler


__all__ = [
    "serialize_tool_result",
    "tool_handler_factory",
]

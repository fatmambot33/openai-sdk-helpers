"""OpenAI response and tool helpers for structured outputs.

This module provides helper functions for creating OpenAI API response formats
and tool definitions from BaseStructure classes. These helpers simplify the
process of configuring structured outputs for both Assistant and chat
completion APIs.
"""

from __future__ import annotations

from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.responses.response_text_config_param import ResponseTextConfigParam

from .base import BaseStructure
from ..utils import log


def assistant_tool_definition(
    structure: type[BaseStructure], name: str, description: str
) -> dict:
    """Build a function tool definition for OpenAI Assistants.

    Creates a tool definition compatible with the Assistant API that uses
    the structure's schema as function parameters.

    Parameters
    ----------
    structure : type[BaseStructure]
        Structure class that defines the tool schema.
    name : str
        Name of the function tool.
    description : str
        Description of what the function tool does.

    Returns
    -------
    dict
        Assistant tool definition payload in OpenAI format.

    Examples
    --------
    >>> from openai_sdk_helpers.structure import BaseStructure
    >>> tool = assistant_tool_definition(
    ...     BaseStructure,
    ...     "process_data",
    ...     "Process input data"
    ... )
    """
    log(f"{structure.__name__}::assistant_tool_definition")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": structure.get_schema(),
        },
    }


def assistant_format(structure: type[BaseStructure]) -> dict:
    """Build a response format definition for OpenAI Assistants.

    Creates a response format specification that instructs the Assistant API
    to return structured output matching the provided schema.

    Parameters
    ----------
    structure : type[BaseStructure]
        Structure class that defines the response schema.

    Returns
    -------
    dict
        Assistant response format definition in OpenAI format.

    Examples
    --------
    >>> format_def = assistant_format(BaseStructure)
    """
    log(f"{structure.__name__}::assistant_format")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": structure.__name__,
            "schema": structure.get_schema(),
        },
    }


def response_tool_definition(
    structure: type[BaseStructure],
    tool_name: str,
    tool_description: str,
) -> dict:
    """Build a tool definition for OpenAI chat completions.

    Creates a function tool definition compatible with the chat completions
    API, using the structure's schema as parameters.

    Parameters
    ----------
    structure : type[BaseStructure]
        Structure class that defines the tool schema.
    tool_name : str
        Name of the function tool.
    tool_description : str
        Description of what the function tool does.

    Returns
    -------
    dict
        Tool definition payload for chat completions API.

    Examples
    --------
    >>> tool = response_tool_definition(
    ...     BaseStructure,
    ...     "analyze",
    ...     "Analyze data"
    ... )
    """
    log(f"{structure.__name__}::response_tool_definition")
    return {
        "type": "function",
        "name": tool_name,
        "description": tool_description,
        "parameters": structure.get_schema(),
        "strict": True,
        "additionalProperties": False,
    }


def response_format(structure: type[BaseStructure]) -> ResponseTextConfigParam:
    """Build a response format for OpenAI chat completions.

    Creates a response format specification that instructs the chat
    completions API to return structured output matching the schema.

    Parameters
    ----------
    structure : type[BaseStructure]
        Structure class that defines the response schema.

    Returns
    -------
    ResponseTextConfigParam
        Response format definition for chat completions API.

    Examples
    --------
    >>> format_spec = response_format(BaseStructure)
    """
    log(f"{structure.__name__}::response_format")
    response_format_text_JSONSchema_config_param = (
        ResponseFormatTextJSONSchemaConfigParam(
            name=structure.__name__,
            schema=structure.get_schema(),
            type="json_schema",
            description="This is a JSON schema format for the output structure.",
            strict=True,
        )
    )
    return ResponseTextConfigParam(format=response_format_text_JSONSchema_config_param)


__all__ = [
    "assistant_tool_definition",
    "assistant_format",
    "response_tool_definition",
    "response_format",
]

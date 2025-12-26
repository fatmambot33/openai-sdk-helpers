"""OpenAI response and tool helpers for structured outputs."""

from __future__ import annotations

from typing import Type

from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.responses.response_text_config_param import ResponseTextConfigParam

from .base import BaseStructure
from ..utils import log


def assistant_tool_definition(
    structure: Type[BaseStructure],
    name: str,
    description: str,
    force_required: bool = False,
) -> dict:
    """Build a function tool definition for OpenAI Assistants.

    Parameters
    ----------
    structure : type[BaseStructure]
        Structure class that defines the tool schema.
    name : str
        Name of the function tool.
    description : str
        Description of what the function tool does.
    force_required : bool, default=False
        When ``True``, mark all object properties as required.

    Returns
    -------
    dict
        Assistant tool definition payload.
    """
    log(f"{structure.__name__}::assistant_tool_definition")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": structure.get_schema(force_required=force_required),
        },
    }


def assistant_format(
    structure: Type[BaseStructure], force_required: bool = False
) -> dict:
    """Build a response format definition for OpenAI Assistants.

    Parameters
    ----------
    structure : type[BaseStructure]
        Structure class that defines the response schema.
    force_required : bool, default=False
        When ``True``, mark all object properties as required.

    Returns
    -------
    dict
        Assistant response format definition.
    """
    log(f"{structure.__name__}::assistant_format")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": structure.__name__,
            "schema": structure.get_schema(force_required=force_required),
        },
    }


def response_tool_definition(
    structure: Type[BaseStructure],
    tool_name: str,
    tool_description: str,
    force_required: bool = False,
) -> dict:
    """Build a tool definition for OpenAI chat completions.

    Parameters
    ----------
    structure : type[BaseStructure]
        Structure class that defines the tool schema.
    tool_name : str
        Name of the function tool.
    tool_description : str
        Description of what the function tool does.
    force_required : bool, default=False
        When ``True``, mark all object properties as required.

    Returns
    -------
    dict
        Tool definition payload for chat completions.
    """
    log(f"{structure.__name__}::response_tool_definition")
    return {
        "type": "function",
        "name": tool_name,
        "description": tool_description,
        "parameters": structure.get_schema(force_required=force_required),
        "strict": True,
        "additionalProperties": False,
    }


def response_format(
    structure: Type[BaseStructure], force_required: bool = False
) -> ResponseTextConfigParam:
    """Build a response format for OpenAI chat completions.

    Parameters
    ----------
    structure : type[BaseStructure]
        Structure class that defines the response schema.
    force_required : bool, default=False
        When ``True``, mark all object properties as required.

    Returns
    -------
    ResponseTextConfigParam
        Response format definition.
    """
    log(f"{structure.__name__}::response_format")
    response_format_text_JSONSchema_config_param = (
        ResponseFormatTextJSONSchemaConfigParam(
            name=structure.__name__,
            schema=structure.get_schema(force_required=force_required),
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

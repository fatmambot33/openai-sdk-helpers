"""Structured output model for prompts.

This module defines a simple Pydantic model for representing prompt text
used in OpenAI API requests.
"""

from __future__ import annotations

from .base import BaseStructure, spec_field


class PromptStructure(BaseStructure):
    """Structured representation of prompt text for OpenAI API requests.

    Simple structure containing a single prompt string with examples.

    Attributes
    ----------
    prompt : str
        The prompt text to use for the OpenAI API request.

    Methods
    -------
    print()
        Return the formatted model fields.

    Examples
    --------
    >>> prompt_struct = PromptStructure(
    ...     prompt="What is the capital of France?"
    ... )
    >>> print(prompt_struct.prompt)
    'What is the capital of France?'
    """

    prompt: str = spec_field(
        "prompt",
        description="The prompt text to use for the OpenAI API request.",
        examples=[
            "What is the capital of France?",
            "Generate a summary of the latest news in AI.",
        ],
    )

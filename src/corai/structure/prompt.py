"""Shared structured output model for prompts."""

from __future__ import annotations

from .base import BaseStructure, spec_field


class PromptStructure(BaseStructure):
    """The prompt text to use for the OpenAI API request.

    Methods
    -------
    print()
        Return the formatted model fields.
    """

    prompt: str = spec_field(
        "prompt",
        description="The prompt text to use for the OpenAI API request.",
        examples=[
            "What is the capital of France?",
            "Generate a summary of the latest news in AI.",
        ],
    )

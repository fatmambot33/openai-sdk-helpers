"""Developer configuration for the example Streamlit chat app."""

from __future__ import annotations

from pydantic import Field

from openai_sdk_helpers.config import OpenAISettings
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure

DEFAULT_MODEL_FALLBACK = "gpt-4o-mini"


class ExampleResponsePayload(BaseStructure):
    """Generic structured payload returned by the assistant.

    Methods
    -------
    None
    """

    title: str = Field(description="Short title for the response.")
    summary: str = Field(description="Concise summary of the assistant's reply.")
    key_points: list[str] | None = Field(
        default=None, description="Optional bullet points or highlights."
    )


class ExampleResponse(ResponseBase[ExampleResponsePayload]):
    """Response tuned for a generic chat experience with structured output.

    Methods
    -------
    __init__()
        Configure a general-purpose response session using OpenAI settings.
    """

    def __init__(self) -> None:
        """Initialize the example response with default OpenAI settings."""
        settings = OpenAISettings.from_env()
        model_name = settings.default_model or DEFAULT_MODEL_FALLBACK
        super().__init__(
            instructions=(
                "You are a concise and helpful assistant. Keep replies brief and "
                "friendly. Populate the structured fields with a title, a short "
                "summary, and any useful key points."
            ),
            tools=[],
            schema=ExampleResponsePayload.response_format(),
            output_structure=ExampleResponsePayload,
            tool_handlers={},
            client=settings.create_client(),
            model=model_name,
        )


APP_CONFIG = {
    "response": ExampleResponse,
    "display_title": "Example assistant chat",
    "description": "Config-driven chat experience for internal demos.",
}

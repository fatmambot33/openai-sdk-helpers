"""Tests for the BaseResponse class."""

from openai_sdk_helpers.response.base import BaseResponse


def test_response_base_initialization(openai_settings, mock_openai_client):
    """Test the initialization of the BaseResponse class."""
    instance = BaseResponse(
        instructions="Test instructions",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    assert instance._instructions == "Test instructions"
    assert instance._model == "gpt-4o-mini"
    assert instance.messages.messages[0].role == "system"
    assert instance._client is mock_openai_client

"""Tests for the BaseResponse class."""

from unittest.mock import MagicMock, patch

import pytest

from openai_sdk_helpers.response.base import BaseResponse


@pytest.fixture
def mock_openai_client():
    """Fixture for a mock OpenAI client."""
    return MagicMock()


def test_response_base_initialization(mock_openai_client):
    """Test the initialization of the BaseResponse class."""
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = mock_openai_client
        instance = BaseResponse(
            instructions="Test instructions",
            tools=[],
            schema=None,
            output_structure=None,
            tool_handlers={},
            model="gpt-4o-mini",
            api_key="test_api_key",
        )
        assert instance._instructions == "Test instructions"
        assert instance._model == "gpt-4o-mini"
        assert instance.messages.messages[0].role == "system"

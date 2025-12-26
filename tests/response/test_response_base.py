"""Tests for the ResponseBase class."""

from unittest.mock import MagicMock, patch

import pytest

from corai.response.base import ResponseBase


@pytest.fixture
def mock_openai_client():
    """Fixture for a mock OpenAI client."""
    return MagicMock()


def test_response_base_initialization(mock_openai_client):
    """Test the initialization of the ResponseBase class."""
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = mock_openai_client
        instance = ResponseBase(
            instructions="Test instructions",
            tools=[],
            schema=None,
            output_structure=None,
            tool_handlers={},
            model="gpt-4",
            api_key="test_api_key",
        )
        assert instance._instructions == "Test instructions"
        assert instance._model == "gpt-4"
        assert instance.messages.messages[0].role == "system"

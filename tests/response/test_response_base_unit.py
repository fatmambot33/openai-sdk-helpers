"""Unit tests for the ResponseBase class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corai.response.base import ResponseBase


@pytest.fixture
def mock_openai_client():
    """Return a mock OpenAI client."""
    return MagicMock()


@pytest.fixture
def response_base(mock_openai_client):
    """Return a ResponseBase instance."""
    return ResponseBase(
        instructions="test instructions",
        tools=[],
        schema=None,
        output_structure=None,
        tool_handlers={},
        client=mock_openai_client,
        model="test_model",
    )


def test_response_base_initialization(response_base):
    """Test ResponseBase initialization."""
    assert response_base._instructions == "test instructions"
    assert response_base._model == "test_model"


def test_data_path(response_base, tmp_path):
    """Test the data_path property."""
    response_base._data_path_fn = lambda module_name: tmp_path / module_name
    response_base._module_name = "test_module"
    assert (
        response_base.data_path
        == tmp_path / "test_module" / "responsebase" / "responsebase"
    )


def test_close(response_base):
    """Test the close method."""
    response_base._user_vector_storage = MagicMock()
    response_base._system_vector_storage = MagicMock()
    response_base.close()
    response_base._user_vector_storage.delete.assert_called_once()
    response_base._system_vector_storage.delete.assert_called_once()


def test_save(response_base, tmp_path):
    """Test the save method."""
    response_base._save_path = tmp_path
    with patch.object(response_base.messages, "to_json_file") as mock_to_json_file:
        response_base.save()
        mock_to_json_file.assert_called_once()

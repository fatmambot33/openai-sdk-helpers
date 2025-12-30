"""Unit tests for the ResponseBase class."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from openai_sdk_helpers.response import attach_vector_store
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.response.messages import ResponseMessage


@pytest.fixture
def mock_openai_client():
    """Return a mock OpenAI client."""
    client = MagicMock()
    client.vector_stores.list.return_value.data = []
    return client


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
    response_base._cleanup_user_vector_storage = True
    response_base._cleanup_system_vector_storage = True
    response_base.close()
    response_base._user_vector_storage.delete.assert_called_once()
    response_base._system_vector_storage.delete.assert_called_once()


def test_close_skips_external_stores(response_base):
    """Ensure externally managed vector stores are preserved."""
    response_base._user_vector_storage = MagicMock()
    response_base._system_vector_storage = MagicMock()
    response_base.close()
    response_base._user_vector_storage.delete.assert_not_called()
    response_base._system_vector_storage.delete.assert_not_called()


def test_save(response_base, tmp_path):
    """Test the save method."""
    response_base._save_path = tmp_path
    with patch.object(response_base.messages, "to_json_file") as mock_to_json_file:
        response_base.save()
        mock_to_json_file.assert_called_once()


def test_attach_vector_store_adds_file_search(response_base):
    """Attach a new vector store when no file_search tool exists."""

    response_base._client.vector_stores.list.return_value.data = [
        SimpleNamespace(id="vs_1", name="store-one"),
    ]

    resolved_ids = attach_vector_store(response_base, "store-one")

    assert resolved_ids == ["vs_1"]
    assert response_base._tools[-1] == {
        "type": "file_search",
        "vector_store_ids": ["vs_1"],
    }


def test_attach_vector_store_merges_ids(response_base):
    """Merge new vector stores into an existing file_search tool."""

    response_base._tools.append(
        {"type": "file_search", "vector_store_ids": ["vs_existing"]}
    )
    response_base._client.vector_stores.list.return_value.data = [
        SimpleNamespace(id="vs_existing", name="existing-store"),
        SimpleNamespace(id="vs_new", name="store-two"),
    ]

    resolved_ids = attach_vector_store(response_base, ["existing-store", "store-two"])

    assert resolved_ids == ["vs_existing", "vs_new"]
    assert response_base._tools[0]["vector_store_ids"] == ["vs_existing", "vs_new"]


def test_attach_vector_store_raises_for_missing_store(response_base):
    """Raise an error when a vector store cannot be resolved."""

    with pytest.raises(ValueError):
        attach_vector_store(response_base, "missing-store")


def test_attach_vector_store_requires_api_key():
    """Raise when no client or API key is available for lookup."""

    response = cast(ResponseBase[Any], SimpleNamespace(_client=None, _tools=[]))

    with pytest.raises(ValueError):
        attach_vector_store(response, "store-one")


def test_get_last_message_returns_latest_assistant(response_base):
    """Return the most recent assistant message when available."""

    response_base.messages.messages.append(ResponseMessage(role="user", content="hi"))
    first_assistant = ResponseMessage(role="assistant", content="first")
    latest_assistant = ResponseMessage(role="assistant", content="second")
    response_base.messages.messages.extend([first_assistant, latest_assistant])

    assert response_base.get_last_message() is latest_assistant


def test_get_last_message_handles_missing_role(response_base):
    """Return None when the requested role is not present."""

    response_base.messages.messages.append(ResponseMessage(role="user", content="hi"))

    assert response_base.get_last_message(role="assistant") is None

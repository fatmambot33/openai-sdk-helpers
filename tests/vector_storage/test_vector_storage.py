"""Tests for the VectorStorage class."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from corai.vector_storage.storage import VectorStorage


@pytest.fixture
def mock_openai_client():
    """Return a mock OpenAI client."""
    client = MagicMock()
    client.vector_stores.list.return_value.data = []
    return client


@pytest.fixture
def vector_storage(mock_openai_client):
    """Return a VectorStorage instance."""
    vs = VectorStorage(
        store_name="test_store", client=mock_openai_client, model="test_model"
    )
    vs._vector_storage.name = "test_store"
    return vs


def test_summarize(vector_storage):
    """Test the summarize method."""
    with pytest.raises(RuntimeError):
        vector_storage.summarize("test_query")


def test_vector_storage_initialization(vector_storage, mock_openai_client):
    """Test VectorStorage initialization."""
    mock_openai_client.vector_stores.create.assert_called_once_with(name="test_store")
    assert vector_storage._vector_storage.name == "test_store"


def test_search(vector_storage, mock_openai_client):
    """Test the search method."""
    vector_storage.search("test_query")
    mock_openai_client.vector_stores.search.assert_called_once_with(
        vector_store_id=vector_storage.id,
        query="test_query",
        max_num_results=5,
    )


def test_id_property(vector_storage):
    """Test the id property."""
    vector_storage._vector_storage.id = "test_id"
    assert vector_storage.id == "test_id"


@patch.object(VectorStorage, "delete_file")
def test_delete(mock_delete_file, vector_storage, mock_openai_client):
    """Test the delete method."""
    vector_storage._existing_files = {"test_file": "test_id"}
    vector_storage.delete()
    mock_delete_file.assert_called_once_with("test_id")
    mock_openai_client.vector_stores.delete.assert_called_once_with(vector_storage.id)

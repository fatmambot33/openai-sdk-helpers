"""Tests for FilesAPIManager."""

from unittest.mock import MagicMock, Mock

import pytest

from openai_sdk_helpers.files_api import FilesAPIManager


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.files = MagicMock()
    return client


@pytest.fixture
def files_manager(mock_client):
    """Create a FilesAPIManager with mock client."""
    return FilesAPIManager(mock_client, auto_track=True)


def test_init(mock_client):
    """Test FilesAPIManager initialization."""
    manager = FilesAPIManager(mock_client, auto_track=True)
    assert manager._client == mock_client
    assert manager._auto_track is True
    assert len(manager.tracked_files) == 0


def test_create_from_path(files_manager, mock_client, tmp_path):
    """Test creating file from path."""
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Mock the API response
    mock_file = Mock()
    mock_file.id = "file-123"
    mock_file.filename = "test.txt"
    mock_client.files.create.return_value = mock_file

    # Create file
    result = files_manager.create(test_file, purpose="user_data")

    # Verify
    assert result.id == "file-123"
    assert "file-123" in files_manager.tracked_files
    mock_client.files.create.assert_called_once()


def test_create_from_string_path(files_manager, mock_client, tmp_path):
    """Test creating file from string path."""
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Mock the API response
    mock_file = Mock()
    mock_file.id = "file-456"
    mock_file.filename = "test.txt"
    mock_client.files.create.return_value = mock_file

    # Create file with string path
    result = files_manager.create(str(test_file), purpose="assistants")

    # Verify
    assert result.id == "file-456"
    assert "file-456" in files_manager.tracked_files


def test_create_without_tracking(files_manager, mock_client, tmp_path):
    """Test creating file without tracking."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    mock_file = Mock()
    mock_file.id = "file-789"
    mock_file.filename = "test.txt"
    mock_client.files.create.return_value = mock_file

    # Create file without tracking
    result = files_manager.create(test_file, purpose="fine-tune", track=False)

    # Verify not tracked
    assert result.id == "file-789"
    assert "file-789" not in files_manager.tracked_files


def test_batch_upload(files_manager, mock_client, tmp_path):
    """Test uploading multiple files."""
    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"
    file_one.write_text("first")
    file_two.write_text("second")

    mock_first = Mock()
    mock_first.id = "file-111"
    mock_first.filename = "one.txt"
    mock_second = Mock()
    mock_second.id = "file-222"
    mock_second.filename = "two.txt"
    mock_client.files.create.side_effect = [mock_first, mock_second]

    results = files_manager.batch_upload(
        [file_one, file_two], purpose="user_data", track=True
    )

    assert [file_obj.id for file_obj in results] == ["file-111", "file-222"]
    assert "file-111" in files_manager.tracked_files
    assert "file-222" in files_manager.tracked_files
    assert mock_client.files.create.call_count == 2


def test_batch_upload_empty(files_manager):
    """Test batch upload with no files."""
    assert files_manager.batch_upload([], purpose="user_data") == []


def test_create_nonexistent_file(files_manager):
    """Test creating from nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        files_manager.create("/nonexistent/file.txt", purpose="user_data")


def test_retrieve(files_manager, mock_client):
    """Test retrieving file info."""
    mock_file = Mock()
    mock_file.id = "file-123"
    mock_client.files.retrieve.return_value = mock_file

    result = files_manager.retrieve("file-123")

    assert result.id == "file-123"
    mock_client.files.retrieve.assert_called_once_with("file-123")


def test_list_all(files_manager, mock_client):
    """Test listing all files."""
    from openai import NOT_GIVEN

    mock_page = Mock()
    mock_client.files.list.return_value = mock_page

    result = files_manager.list()

    assert result == mock_page
    mock_client.files.list.assert_called_once_with(limit=NOT_GIVEN)


def test_list_with_purpose(files_manager, mock_client):
    """Test listing files filtered by purpose."""
    mock_page = Mock()
    mock_client.files.list.return_value = mock_page

    result = files_manager.list(purpose="user_data", limit=10)

    assert result == mock_page
    mock_client.files.list.assert_called_once_with(purpose="user_data", limit=10)


def test_delete(files_manager, mock_client):
    """Test deleting a file."""
    # Add file to tracked files
    mock_file = Mock()
    mock_file.id = "file-123"
    files_manager.tracked_files["file-123"] = mock_file

    # Mock delete response
    mock_deleted = Mock()
    mock_deleted.deleted = True
    mock_client.files.delete.return_value = mock_deleted

    # Delete file
    result = files_manager.delete("file-123")

    # Verify
    assert result.deleted is True
    assert "file-123" not in files_manager.tracked_files
    mock_client.files.delete.assert_called_once_with("file-123")


def test_delete_without_untrack(files_manager, mock_client):
    """Test deleting without untracking."""
    files_manager.tracked_files["file-123"] = Mock()

    mock_deleted = Mock()
    mock_deleted.deleted = True
    mock_client.files.delete.return_value = mock_deleted

    result = files_manager.delete("file-123", untrack=False)

    assert result.deleted is True
    assert "file-123" in files_manager.tracked_files


def test_retrieve_content(files_manager, mock_client):
    """Test retrieving file content."""
    mock_content = Mock()
    mock_content.read.return_value = b"file content"
    mock_client.files.content.return_value = mock_content

    content = files_manager.retrieve_content("file-123")

    assert content == b"file content"
    mock_client.files.content.assert_called_once_with("file-123")


def test_cleanup(files_manager, mock_client):
    """Test cleanup of tracked files."""
    # Add tracked files
    for i in range(3):
        files_manager.tracked_files[f"file-{i}"] = Mock()

    # Mock delete responses
    mock_deleted = Mock()
    mock_deleted.deleted = True
    mock_client.files.delete.return_value = mock_deleted

    # Cleanup
    results = files_manager.cleanup()

    # Verify all deleted
    assert len(results) == 3
    assert all(results.values())
    assert len(files_manager.tracked_files) == 0
    assert mock_client.files.delete.call_count == 3


def test_cleanup_with_errors(files_manager, mock_client):
    """Test cleanup handles errors gracefully."""
    files_manager.tracked_files["file-1"] = Mock()
    files_manager.tracked_files["file-2"] = Mock()

    # First succeeds, second fails
    mock_client.files.delete.side_effect = [
        Mock(deleted=True),
        Exception("API error"),
    ]

    results = files_manager.cleanup()

    assert len(results) == 2
    assert results["file-1"] is True
    assert results["file-2"] is False


def test_context_manager(mock_client, tmp_path):
    """Test context manager automatically cleans up."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    mock_file = Mock()
    mock_file.id = "file-123"
    mock_file.filename = "test.txt"
    mock_client.files.create.return_value = mock_file

    mock_deleted = Mock()
    mock_deleted.deleted = True
    mock_client.files.delete.return_value = mock_deleted

    with FilesAPIManager(mock_client) as manager:
        manager.create(test_file, purpose="user_data")
        assert len(manager.tracked_files) == 1

    # Should be cleaned up after context
    mock_client.files.delete.assert_called_once_with("file-123")


def test_len(files_manager):
    """Test __len__ returns tracked file count."""
    assert len(files_manager) == 0

    files_manager.tracked_files["file-1"] = Mock()
    files_manager.tracked_files["file-2"] = Mock()

    assert len(files_manager) == 2


def test_repr(files_manager):
    """Test __repr__ shows tracked count."""
    files_manager.tracked_files["file-1"] = Mock()
    files_manager.tracked_files["file-2"] = Mock()

    repr_str = repr(files_manager)
    assert "FilesAPIManager" in repr_str
    assert "2" in repr_str


def test_create_with_expires_after(files_manager, mock_client, tmp_path):
    """Test creating file with expires_after parameter."""
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Mock the API response
    mock_file = Mock()
    mock_file.id = "file-123"
    mock_file.filename = "test.txt"
    mock_client.files.create.return_value = mock_file

    # Create file with 24h expiration (86400 seconds)
    result = files_manager.create(test_file, purpose="user_data", expires_after=86400)

    # Verify the expires_after was converted to OpenAI API format
    call_kwargs = mock_client.files.create.call_args[1]
    assert "expires_after" in call_kwargs
    expires_dict = call_kwargs["expires_after"]
    assert expires_dict["anchor"] == "created_at"
    assert expires_dict["seconds"] == 86400  # 86400 seconds = 24 hours

    # Verify result
    assert result.id == "file-123"
    assert "file-123" in files_manager.tracked_files


def test_create_with_expires_after_hours_conversion(
    files_manager, mock_client, tmp_path
):
    """Test expires_after conversion for hour-based values."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    mock_file = Mock()
    mock_file.id = "file-456"
    mock_client.files.create.return_value = mock_file

    # Create file with 1 hour expiration (3600 seconds)
    result = files_manager.create(test_file, purpose="user_data", expires_after=3600)

    # Verify conversion to OpenAI API format
    call_kwargs = mock_client.files.create.call_args[1]
    expires_dict = call_kwargs["expires_after"]
    assert expires_dict["anchor"] == "created_at"
    assert expires_dict["seconds"] == 3600  # 3600 seconds = 1 hour

    assert result.id == "file-456"


def test_create_without_expires_after(files_manager, mock_client, tmp_path):
    """Test creating file without expires_after uses no expiration."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    mock_file = Mock()
    mock_file.id = "file-789"
    mock_client.files.create.return_value = mock_file

    # Create file without expiration
    result = files_manager.create(test_file, purpose="fine-tune")

    # Verify no expires_after was passed
    call_kwargs = mock_client.files.create.call_args[1]
    assert "expires_after" not in call_kwargs

    assert result.id == "file-789"

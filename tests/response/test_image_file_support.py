"""Tests for ResponseBase with image and file data support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openai_sdk_helpers.response.base import ResponseBase


@pytest.fixture
def response_base(openai_settings, tmp_path):
    """Return a ResponseBase instance for testing."""
    return ResponseBase(
        name="test",
        instructions="test instructions",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
        data_path=tmp_path,
    )


def test_build_input_with_images(response_base, tmp_path):
    """Test _build_input with image attachments."""
    # Create a temporary image file
    image_path = tmp_path / "test_image.jpg"
    image_content = b"fake image content"
    image_path.write_bytes(image_content)

    # Build input with image (automatically detected)
    response_base._build_input(content="What's in this image?", files=[str(image_path)])

    # Verify message was added
    assert len(response_base.messages.messages) == 2  # system + user
    user_message = response_base.messages.messages[-1]
    assert user_message.role == "user"

    # Check that the message content has an image
    content = user_message.content["content"]
    assert len(content) == 2  # text + image
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert "base64" in content[1]["image_url"]


def test_build_input_with_file_data(response_base, tmp_path):
    """Test _build_input with file data attachments."""
    # Create a temporary file
    file_path = tmp_path / "test_file.pdf"
    file_content = b"fake pdf content"
    file_path.write_bytes(file_content)

    # Build input with file data (automatically detected as non-image)
    response_base._build_input(content="Analyze this document", files=[str(file_path)])

    # Verify message was added
    assert len(response_base.messages.messages) == 2  # system + user
    user_message = response_base.messages.messages[-1]
    assert user_message.role == "user"

    # Check that the message content has a file
    content = user_message.content["content"]
    assert len(content) == 2  # text + file
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_file"
    assert "file_data" in content[1]
    assert "base64" in content[1]["file_data"]
    assert content[1]["filename"] == "test_file.pdf"


def test_build_input_with_base64_attachments(response_base, tmp_path):
    """Test _build_input without vector store flag (default is inline base64)."""
    # Create a temporary file
    file_path = tmp_path / "test_file.pdf"
    file_content = b"test content"
    file_path.write_bytes(file_content)

    # Build input - non-image files default to base64
    response_base._build_input(content="Process this file", files=[str(file_path)])

    # Verify message was added
    assert len(response_base.messages.messages) == 2  # system + user
    user_message = response_base.messages.messages[-1]
    assert user_message.role == "user"

    # Check that the message content has a file with base64 data
    content = user_message.content["content"]
    assert len(content) == 2  # text + file
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_file"
    assert "file_data" in content[1]
    assert "base64" in content[1]["file_data"]


def test_build_input_with_multiple_types(response_base, tmp_path):
    """Test _build_input with multiple attachment types."""
    # Create temporary files
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(b"fake image")

    file_path = tmp_path / "test_doc.pdf"
    file_path.write_bytes(b"fake pdf")

    # Build input with multiple types - automatic detection
    response_base._build_input(
        content="Analyze these", files=[str(image_path), str(file_path)]
    )

    # Verify message was added
    user_message = response_base.messages.messages[-1]
    content = user_message.content["content"]

    # Should have text + file + image (order: files first, then images)
    assert len(content) == 3
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_file"
    assert content[2]["type"] == "input_image"


def test_build_input_with_multiple_messages_attaches_once(response_base, tmp_path):
    """Test _build_input attaches files only to the first message."""
    file_path = tmp_path / "test_doc.pdf"
    file_path.write_bytes(b"fake pdf")

    response_base._build_input(
        content=["First message", "Second message"],
        files=[str(file_path)],
    )

    assert len(response_base.messages.messages) == 3  # system + 2 user
    first_message = response_base.messages.messages[-2]
    second_message = response_base.messages.messages[-1]

    first_content = first_message.content["content"]
    second_content = second_message.content["content"]

    assert len(first_content) == 2  # text + file
    assert first_content[1]["type"] == "input_file"
    assert len(second_content) == 1  # text only
    assert second_content[0]["type"] == "input_text"


def test_build_input_vector_store_still_works(response_base, tmp_path):
    """Test that vector store attachment still works with use_vector_store flag."""
    # Create a temporary file
    file_path = tmp_path / "test_file.txt"
    file_path.write_bytes(b"test content")

    # Mock the vector storage
    mock_storage = MagicMock()
    mock_file = MagicMock()
    mock_file.id = "file_123"
    mock_storage.upload_file.return_value = mock_file
    mock_storage.id = "vs_123"

    with patch(
        "openai_sdk_helpers.vector_storage.VectorStorage", return_value=mock_storage
    ):
        response_base._build_input(
            content="Process this file",
            files=[str(file_path)],
            use_vector_store=True,
        )

    # Verify vector storage was used
    mock_storage.upload_file.assert_called_once()

    # Verify message uses file_id
    user_message = response_base.messages.messages[-1]
    content = user_message.content["content"]
    assert content[1]["type"] == "input_file"
    assert content[1]["file_id"] == "file_123"


def test_run_sync_with_files(response_base, tmp_path):
    """Test run_sync method signature accepts files parameter."""
    # We just need to verify the method signature accepts the parameter
    from inspect import signature

    sig = signature(response_base.run_sync)
    assert "files" in sig.parameters
    assert "use_vector_store" in sig.parameters


def test_run_async_with_files(response_base, tmp_path):
    """Test run_async method signature accepts files parameter."""
    # We just need to verify the method signature accepts the parameter
    from inspect import signature

    sig = signature(response_base.run_async)
    assert "files" in sig.parameters
    assert "use_vector_store" in sig.parameters

"""Tests for Agents SDK file helper utilities."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openai_sdk_helpers.agent.configuration import AgentConfiguration
from openai_sdk_helpers.agent.files import build_agent_input_messages
from openai_sdk_helpers.settings import OpenAISettings


def test_build_agent_input_messages_with_image(tmp_path):
    """Ensure images are encoded as input_image entries."""
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image data")

    messages = build_agent_input_messages(
        "What is in this image?", files=str(image_path)
    )

    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/")


def test_build_agent_input_messages_requires_manager_for_documents(tmp_path):
    """Ensure document uploads require a files manager."""
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"fake pdf")

    with pytest.raises(ValueError, match="files_manager"):
        build_agent_input_messages("Summarize this", files=str(file_path))


def test_build_agent_input_messages_uploads_documents(tmp_path):
    """Ensure documents are uploaded via the provided files manager."""
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"fake pdf")

    files_manager = MagicMock()
    files_manager.batch_upload.return_value = [SimpleNamespace(id="file_123")]

    messages = build_agent_input_messages(
        "Summarize this",
        files=[str(file_path)],
        files_manager=files_manager,
    )

    files_manager.batch_upload.assert_called_once_with(
        [str(file_path)],
        purpose="user_data",
        expires_after=86400,
    )
    content = messages[0]["content"]
    assert content[1]["type"] == "input_file"
    assert content[1]["file_id"] == "file_123"


def test_build_agent_input_messages_creates_manager_from_settings(
    tmp_path, monkeypatch
):
    """Ensure settings can create a FilesAPIManager for uploads."""
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"fake pdf")

    mock_manager = MagicMock()
    mock_manager.batch_upload.return_value = [SimpleNamespace(id="file_456")]
    monkeypatch.setattr(
        "openai_sdk_helpers.agent.files.FilesAPIManager",
        MagicMock(return_value=mock_manager),
    )

    monkeypatch.setattr(
        OpenAISettings, "create_client", MagicMock(return_value=MagicMock())
    )
    settings = OpenAISettings(api_key="test_api_key", default_model="gpt-4o-mini")

    messages = build_agent_input_messages(
        "Summarize this",
        files=[str(file_path)],
        openai_settings=settings,
    )

    mock_manager.batch_upload.assert_called_once_with(
        [str(file_path)],
        purpose="user_data",
        expires_after=86400,
    )
    content = messages[0]["content"]
    assert content[1]["type"] == "input_file"
    assert content[1]["file_id"] == "file_456"


def test_build_agent_input_messages_attaches_once(tmp_path):
    """Ensure attachments are only added to the first message."""
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image data")

    messages = build_agent_input_messages(
        ["First message", "Second message"],
        files=[str(image_path)],
    )

    assert len(messages) == 2
    first_content = messages[0]["content"]
    second_content = messages[1]["content"]

    assert len(first_content) == 2
    assert first_content[1]["type"] == "input_image"
    assert len(second_content) == 1
    assert second_content[0]["type"] == "input_text"


def test_agent_configuration_to_openai_settings():
    """Ensure AgentConfiguration builds OpenAISettings with defaults."""
    configuration = AgentConfiguration(
        name="summarizer",
        instructions="Summarize text",
        model="gpt-4o-mini",
    )

    settings = configuration.to_openai_settings(api_key="test_api_key")

    assert settings.api_key == "test_api_key"
    assert settings.default_model == "gpt-4o-mini"

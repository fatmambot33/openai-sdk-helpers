"""Shared fixtures for response tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openai_sdk_helpers.config import OpenAISettings


@pytest.fixture
def mock_openai_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide a mock OpenAI client used by BaseResponse instances."""
    client = MagicMock()
    client.api_key = "test_api_key"
    client.vector_stores.list.return_value.data = []
    monkeypatch.setattr(
        "openai_sdk_helpers.config.OpenAI",
        MagicMock(return_value=client),
    )
    return client


@pytest.fixture
def openai_settings(mock_openai_client: MagicMock) -> OpenAISettings:
    """Return OpenAI settings configured for tests."""
    return OpenAISettings(api_key="test_api_key", default_model="gpt-4o-mini")

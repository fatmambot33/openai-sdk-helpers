"""Tests for build_openai_settings helper function."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openai_sdk_helpers.utils import build_openai_settings


def test_build_openai_settings_from_params(monkeypatch):
    """Test building settings from explicit parameters."""
    # Set a dummy API key to avoid environment lookup
    result = build_openai_settings(
        api_key="sk-test-key-123",
        default_model="gpt-4o",
        timeout=30.0,
        max_retries=3,
    )

    assert result.api_key == "sk-test-key-123"
    assert result.default_model == "gpt-4o"
    assert result.timeout == 30.0
    assert result.max_retries == 3


def test_build_openai_settings_from_env(monkeypatch):
    """Test building settings from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-456")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
    monkeypatch.setenv("OPENAI_TIMEOUT", "60")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")

    result = build_openai_settings()

    assert result.api_key == "sk-env-key-456"
    assert result.default_model == "gpt-4"
    assert result.timeout == 60.0
    assert result.max_retries == 5


def test_build_openai_settings_params_override_env(monkeypatch):
    """Test that explicit parameters override environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5")

    result = build_openai_settings(
        api_key="sk-param-key", default_model="gpt-4o"
    )

    # Parameters should override environment
    assert result.api_key == "sk-param-key"
    assert result.default_model == "gpt-4o"


def test_build_openai_settings_timeout_string_parsing():
    """Test that timeout strings are parsed correctly."""
    result = build_openai_settings(
        api_key="sk-test-key",
        timeout="45.5",  # String that should be parsed to float
    )

    assert result.timeout == 45.5
    assert isinstance(result.timeout, float)


def test_build_openai_settings_max_retries_string_parsing():
    """Test that max_retries strings are parsed correctly."""
    result = build_openai_settings(
        api_key="sk-test-key",
        max_retries="10",  # String that should be parsed to int
    )

    assert result.max_retries == 10
    assert isinstance(result.max_retries, int)


def test_build_openai_settings_invalid_timeout_raises():
    """Test that invalid timeout values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid timeout value"):
        build_openai_settings(
            api_key="sk-test-key",
            timeout="not-a-number",
        )


def test_build_openai_settings_invalid_max_retries_raises():
    """Test that invalid max_retries values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid max_retries value"):
        build_openai_settings(
            api_key="sk-test-key",
            max_retries="not-a-number",
        )


def test_build_openai_settings_missing_api_key_raises(monkeypatch):
    """Test that missing API key raises ValueError."""
    # Clear any existing API key from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Failed to build OpenAI settings"):
        build_openai_settings()


def test_build_openai_settings_extra_kwargs():
    """Test that extra_kwargs are passed through."""
    result = build_openai_settings(
        api_key="sk-test-key",
        custom_header="value",
        another_option=True,
    )

    assert result.extra_client_kwargs.get("custom_header") == "value"
    assert result.extra_client_kwargs.get("another_option") is True


def test_build_openai_settings_all_params():
    """Test building with all parameters specified."""
    result = build_openai_settings(
        api_key="sk-complete-key",
        org_id="org-123",
        project_id="proj-456",
        base_url="https://custom.api.com",
        default_model="gpt-4o-mini",
        timeout=120.0,
        max_retries=5,
    )

    assert result.api_key == "sk-complete-key"
    assert result.org_id == "org-123"
    assert result.project_id == "proj-456"
    assert result.base_url == "https://custom.api.com"
    assert result.default_model == "gpt-4o-mini"
    assert result.timeout == 120.0
    assert result.max_retries == 5


def test_build_openai_settings_with_dotenv(tmp_path, monkeypatch):
    """Test loading from a custom .env file."""
    # Clear environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Create a .env file
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-dotenv-key\n"
        "OPENAI_MODEL=gpt-4\n"
        "OPENAI_TIMEOUT=90\n"
    )

    result = build_openai_settings(dotenv_path=env_file)

    assert result.api_key == "sk-dotenv-key"
    assert result.default_model == "gpt-4"
    assert result.timeout == 90.0


def test_build_openai_settings_numeric_timeout():
    """Test that numeric timeout values work correctly."""
    result = build_openai_settings(
        api_key="sk-test-key",
        timeout=75.5,
    )

    assert result.timeout == 75.5


def test_build_openai_settings_numeric_max_retries():
    """Test that numeric max_retries values work correctly."""
    result = build_openai_settings(
        api_key="sk-test-key",
        max_retries=7,
    )

    assert result.max_retries == 7

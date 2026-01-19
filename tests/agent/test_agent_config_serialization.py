"""Tests for AgentConfiguration JSON serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openai_sdk_helpers.agent.configuration import AgentConfiguration


def test_agent_config_to_json() -> None:
    """Test that AgentConfiguration can be serialized to JSON."""
    configuration = AgentConfiguration(
        name="test_agent",
        instructions="Test instructions",
        description="A test agent",
        model="gpt-4o-mini",
        template_path="/tmp/template.jinja",
    )

    json_data = configuration.to_json()

    assert isinstance(json_data, dict)
    assert json_data["name"] == "test_agent"
    assert json_data["description"] == "A test agent"
    assert json_data["model"] == "gpt-4o-mini"
    assert json_data["template_path"] == "/tmp/template.jinja"
    assert json_data["instructions"] == "Test instructions"
    assert json_data["tools"] is None
    assert json_data["add_web_search_tool"] is False


def test_agent_config_to_json_file(tmp_path: Path) -> None:
    """Test that AgentConfiguration can be written to a JSON file."""
    configuration = AgentConfiguration(
        name="test_agent",
        instructions="Test instructions",
        description="A test agent",
        model="gpt-4o-mini",
    )

    json_file = tmp_path / "agent_config.json"
    result_path = configuration.to_json_file(json_file)

    assert result_path == str(json_file)
    assert json_file.exists()

    # Verify the file content
    with open(json_file, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    assert loaded_data["name"] == "test_agent"
    assert loaded_data["description"] == "A test agent"
    assert loaded_data["model"] == "gpt-4o-mini"


def test_agent_config_json_serialization_with_none_fields() -> None:
    """Test JSON serialization with None fields."""
    configuration = AgentConfiguration(
        name="minimal_agent",
        instructions="Test instructions",
        model="gpt-4o-mini",
    )

    json_data = configuration.to_json()

    assert json_data["name"] == "minimal_agent"
    assert json_data["model"] == "gpt-4o-mini"
    assert json_data["description"] is None
    assert json_data["template_path"] is None
    assert json_data["instructions"] == "Test instructions"
    assert json_data["tools"] is None
    assert json_data["handoffs"] is None
    assert json_data["input_guardrails"] is None
    assert json_data["output_guardrails"] is None
    assert json_data["session"] is None
    assert json_data["add_web_search_tool"] is False


def test_agent_config_from_json() -> None:
    """Test that AgentConfiguration can be deserialized from JSON."""
    json_data = {
        "name": "test_agent",
        "description": "A test agent",
        "model": "gpt-4o-mini",
        "instructions": "You are a helpful assistant",
        "template_path": "/tmp/template.jinja",
        "input_structure": None,
        "output_structure": None,
        "tools": None,
        "model_settings": None,
        "handoffs": None,
        "input_guardrails": None,
        "output_guardrails": None,
        "session": None,
    }

    configuration = AgentConfiguration.from_json(json_data)

    assert configuration.name == "test_agent"
    assert configuration.description == "A test agent"
    assert configuration.model == "gpt-4o-mini"
    assert configuration.instructions == "You are a helpful assistant"
    # template_path gets converted to Path when field name contains "path"
    assert str(configuration.template_path) == "/tmp/template.jinja"


def test_agent_config_from_json_file(tmp_path: Path) -> None:
    """Test that AgentConfiguration can be loaded from a JSON file."""
    json_data = {
        "name": "test_agent",
        "description": "A test agent",
        "model": "gpt-4o-mini",
        "instructions": "Test instructions",
        "template_path": None,
        "input_structure": None,
        "output_structure": None,
        "tools": None,
        "model_settings": None,
        "handoffs": None,
        "input_guardrails": None,
        "output_guardrails": None,
        "session": None,
    }

    json_file = tmp_path / "agent_config.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    configuration = AgentConfiguration.from_json_file(json_file)

    assert configuration.name == "test_agent"
    assert configuration.description == "A test agent"
    assert configuration.model == "gpt-4o-mini"


def test_agent_config_round_trip_serialization() -> None:
    """Test that serialization and deserialization preserve data."""
    original_config = AgentConfiguration(
        name="round_trip_test",
        description="Round trip test agent",
        model="gpt-4o-mini",
        instructions="Test instructions",
    )

    # Serialize to JSON
    json_data = original_config.to_json()

    # Deserialize from JSON
    restored_config = AgentConfiguration.from_json(json_data)

    # Verify all fields match
    assert restored_config.name == original_config.name
    assert restored_config.description == original_config.description
    assert restored_config.model == original_config.model
    assert restored_config.instructions == original_config.instructions
    assert restored_config.template_path == original_config.template_path
    assert restored_config.tools == original_config.tools


def test_agent_config_from_json_file_not_found() -> None:
    """Test that from_json_file raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        AgentConfiguration.from_json_file("/nonexistent/file.json")


def test_agent_config_validation_empty_name() -> None:
    """Test that AgentConfiguration raises TypeError for empty name."""
    with pytest.raises(
        TypeError, match="AgentConfiguration.name must be a non-empty str"
    ):
        AgentConfiguration(name="", instructions="Test", model="gpt-4o-mini")


def test_agent_config_validation_empty_instructions() -> None:
    """Test that AgentConfiguration raises ValueError for empty instructions."""
    with pytest.raises(
        ValueError, match="AgentConfiguration.instructions must be a non-empty str"
    ):
        AgentConfiguration(name="test", model="gpt-4o-mini", instructions="   ")


def test_agent_config_with_path_template(tmp_path: Path) -> None:
    """Test that AgentConfiguration serializes Path objects correctly."""
    template_path = tmp_path / "template.jinja"
    template_path.write_text("Test template")

    configuration = AgentConfiguration(
        name="test_agent",
        instructions="Test instructions",
        model="gpt-4o-mini",
        template_path=template_path,
    )

    json_data = configuration.to_json()

    # Path should be converted to string
    assert isinstance(json_data["template_path"], str)
    assert json_data["template_path"] == str(template_path)

    # Deserialize and verify
    restored_config = AgentConfiguration.from_json(json_data)
    assert isinstance(restored_config.template_path, (str, Path))
    assert str(restored_config.template_path) == str(template_path)

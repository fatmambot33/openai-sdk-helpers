"""Tests for the BaseStructure class."""

from enum import Enum
from typing import List, Optional

import pytest
from pydantic import Field
from pydantic.fields import FieldInfo

from openai_sdk_helpers.structure.base import BaseStructure, SchemaOptions, spec_field
from openai_sdk_helpers.structure.responses import (
    assistant_format,
    assistant_tool_definition,
    response_format,
    response_tool_definition,
)


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class DummyStructure(BaseStructure):
    """A dummy structure for testing."""

    name: str = Field(..., description="The name of the item.")
    age: Optional[int] = Field(None, description="The age of the item.")
    color: Optional[Color] = Field(None, description="The color of the item.")
    tags: Optional[List[str]] = Field(None, description="A list of tags.")


def test_get_prompt():
    """Test the get_prompt method."""
    prompt = DummyStructure.get_prompt()
    assert "# Output Format" in prompt
    assert "- **Name**: The name of the item." in prompt
    assert "- **Age**: Provide the relevant Age." in prompt
    assert "- **Color**: The color of the item." in prompt
    assert "Choose from:" in prompt
    assert "RED: red" in prompt


def test_get_prompt_no_enum_values():
    """Test the get_prompt method without enum values."""
    prompt = DummyStructure.get_prompt(add_enum_values=False)
    assert "Choose from:" not in prompt


def test_get_schema():
    """Test the get_schema method."""
    schema = DummyStructure.get_schema()
    assert schema["title"] == "OutputStructure"
    properties = schema["properties"]
    assert "name" in properties
    assert "age" in properties
    assert "color" in properties
    assert "tags" in properties
    assert properties["name"]["type"] == "string"

    # Check optional int schema
    age_schema = properties["age"]
    assert "anyOf" in age_schema
    assert {"type": "integer"} in age_schema["anyOf"]
    assert {"type": "null"} in age_schema["anyOf"]

    # Check optional enum schema
    color_schema = properties["color"]
    assert "anyOf" in color_schema
    assert {"$ref": "#/$defs/Color"} in color_schema["anyOf"]
    assert {"type": "null"} in color_schema["anyOf"]


def test_get_schema_force_required():
    """Test the get_schema method with force_required."""
    schema = DummyStructure.get_schema()
    assert "required" in schema
    assert "name" in schema["required"]
    assert "age" in schema["required"]
    assert "color" in schema["required"]


class NullOptInStructure(BaseStructure):
    """Structure that opts fields into explicit null entries."""

    headline: str | None = None


def test_get_schema_with_nullable_default():
    """Test get_schema marks ``None`` defaults as explicitly nullable."""

    schema = NullOptInStructure.get_schema()
    properties = schema["properties"]
    headline_schema = properties["headline"]

    any_of = headline_schema.get("anyOf")
    if isinstance(any_of, list):
        assert {"type": "null"} in any_of
    else:
        assert "null" in headline_schema["type"]
    assert "required" in schema and "headline" in schema["required"]


def test_convenience_wrappers_for_response_helpers():
    """Ensure BaseStructure wraps the response helper utilities."""

    assistant_tool = DummyStructure.assistant_tool_definition("demo", "desc")
    assert assistant_tool == assistant_tool_definition(DummyStructure, "demo", "desc")

    assistant_schema = DummyStructure.assistant_format()
    assert assistant_schema == assistant_format(DummyStructure)

    completion_tool = DummyStructure.response_tool_definition("demo", "desc")
    assert completion_tool == response_tool_definition(DummyStructure, "demo", "desc")

    completion_format = DummyStructure.response_format()
    expected_format = response_format(DummyStructure)
    assert completion_format == expected_format


def test_to_json():
    """Test the to_json method."""
    instance = DummyStructure(name="Test", age=42, color=Color.RED, tags=["a", "b"])
    json_data = instance.to_json()
    assert json_data["name"] == "Test"
    assert json_data["age"] == 42
    assert json_data["color"] == "red"
    assert json_data["tags"] == ["a", "b"]


def test_schema_options():
    """Test the SchemaOptions class."""
    options = SchemaOptions(force_required=True)
    assert options.to_kwargs() == {"force_required": True}


def test_spec_field():
    """Test the spec_field function."""
    field = spec_field("test_field", description="A test field.")
    assert isinstance(field, FieldInfo)
    assert field.title == "Test Field"
    assert field.description == "A test field. Return null if none apply."
    assert field.default is None


def test_spec_field_allow_null_false():
    """Ensure spec_field can opt out of null defaults."""

    field = spec_field("required_field", allow_null=False)
    assert field.is_required()
    assert field.description is None


def test_from_raw_input(caplog):
    """Test the from_raw_input method."""
    # Test with valid string enum value
    data = {"name": "Test", "age": 42, "color": "red"}
    instance = DummyStructure.from_raw_input(data)
    assert instance.name == "Test"
    assert instance.age == 42
    assert instance.color == Color.RED

    # Test with invalid enum value
    data = {"name": "Test", "age": 42, "color": "purple"}
    instance = DummyStructure.from_raw_input(data)
    assert instance.color is None
    assert "Invalid value for 'color'" in caplog.text

    # Test with a list of enums (not in DummyStructure, so we need a new class)
    class MultiColorStructure(BaseStructure):
        colors: List[Color]

    data = {"colors": ["red", "blue"]}
    instance = MultiColorStructure.from_raw_input(data)
    assert instance.colors == [Color.RED, Color.BLUE]

    # Test with a mix of valid and invalid enum values in a list
    data = {"colors": ["red", "yellow", "green"]}
    instance = MultiColorStructure.from_raw_input(data)
    assert instance.colors == [Color.RED, Color.GREEN]
    assert "Skipping invalid value for 'colors'" in caplog.text

    # Test with pre-converted enum
    data = {"name": "Test", "age": 42, "color": Color.GREEN}
    instance = DummyStructure.from_raw_input(data)
    assert instance.color == Color.GREEN


def test_save_schema_to_file(tmp_path):
    """Test the save_schema_to_file method."""
    DummyStructure.DATA_PATH = tmp_path
    schema_path = DummyStructure.save_schema_to_file()
    assert schema_path.exists()
    assert schema_path.name == "DummyStructure_schema.json"
    with open(schema_path, "r") as f:
        import json

        schema_data = json.load(f)
    assert schema_data["title"] == "OutputStructure"

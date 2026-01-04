"""Tests for tool handler utilities."""

from __future__ import annotations

import json
import pytest
from pydantic import BaseModel, ValidationError

from openai_sdk_helpers.tools import serialize_tool_result, tool_handler_factory
from openai_sdk_helpers.response.tool_call import parse_tool_arguments


class SampleInput(BaseModel):
    """Sample Pydantic model for testing."""

    query: str
    limit: int = 10


class SampleOutput(BaseModel):
    """Sample output model."""

    results: list[str]
    count: int


def test_serialize_tool_result_with_pydantic():
    """Test serialization of Pydantic models."""
    output = SampleOutput(results=["result1", "result2"], count=2)
    serialized = serialize_tool_result(output)

    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed["results"] == ["result1", "result2"]
    assert parsed["count"] == 2


def test_serialize_tool_result_with_list():
    """Test serialization of lists."""
    result = ["item1", "item2", "item3"]
    serialized = serialize_tool_result(result)

    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed == result


def test_serialize_tool_result_with_dict():
    """Test serialization of dictionaries."""
    result = {"key": "value", "number": 42}
    serialized = serialize_tool_result(result)

    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed == result


def test_serialize_tool_result_with_string():
    """Test serialization of plain strings."""
    result = "plain text result"
    serialized = serialize_tool_result(result)

    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed == result


def test_serialize_tool_result_with_primitives():
    """Test serialization of primitive types."""
    assert serialize_tool_result(42) == "42"
    assert serialize_tool_result(3.14) == "3.14"
    assert serialize_tool_result(True) == "true"
    assert serialize_tool_result(None) == "null"


def test_parse_tool_arguments_with_tool_name():
    """Test enhanced parse_tool_arguments with tool name."""
    args = '{"key": "value"}'
    result = parse_tool_arguments(args, tool_name="test_tool")
    assert result == {"key": "value"}


def test_parse_tool_arguments_error_includes_tool_name():
    """Test that parse errors include tool name for context."""
    invalid_args = '{"key": invalid}'

    with pytest.raises(ValueError) as exc_info:
        parse_tool_arguments(invalid_args, tool_name="my_tool")

    error_msg = str(exc_info.value)
    assert "my_tool" in error_msg
    assert "Raw payload" in error_msg


def test_parse_tool_arguments_truncates_long_payload():
    """Test that long payloads are truncated in error messages."""
    # Create an invalid payload longer than 100 characters
    long_payload = '{"key": invalid_value_' + "x" * 200 + '}'

    with pytest.raises(ValueError) as exc_info:
        parse_tool_arguments(long_payload, tool_name="test")

    error_msg = str(exc_info.value)
    assert "..." in error_msg  # Should be truncated


class MockToolCall:
    """Mock tool call object for testing."""

    def __init__(self, arguments: str, name: str = "test_tool"):
        self.arguments = arguments
        self.name = name


def test_tool_handler_factory_basic():
    """Test basic tool handler without validation."""

    def simple_tool(query: str, limit: int = 10):
        return {"query": query, "limit": limit}

    handler = tool_handler_factory(simple_tool)

    tool_call = MockToolCall('{"query": "test", "limit": 5}')
    result = handler(tool_call)

    # Result should be JSON string
    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert parsed["limit"] == 5


def test_tool_handler_factory_with_validation():
    """Test tool handler with Pydantic validation."""

    def search_tool(query: str, limit: int = 10):
        return SampleOutput(results=[f"result for {query}"], count=1)

    handler = tool_handler_factory(search_tool, input_model=SampleInput)

    tool_call = MockToolCall('{"query": "test search", "limit": 20}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["results"] == ["result for test search"]
    assert parsed["count"] == 1


def test_tool_handler_factory_validation_failure():
    """Test that validation errors are raised with context."""

    def dummy_tool(query: str, limit: int):
        return {}

    handler = tool_handler_factory(dummy_tool, input_model=SampleInput)

    # Missing required field 'query'
    tool_call = MockToolCall('{"limit": 10}', name="search")

    with pytest.raises(ValidationError):
        handler(tool_call)


def test_tool_handler_factory_with_defaults():
    """Test that default values work correctly."""

    def tool_with_defaults(query: str, limit: int = 10, offset: int = 0):
        return {"query": query, "limit": limit, "offset": offset}

    handler = tool_handler_factory(tool_with_defaults)

    # Only provide query, use defaults
    tool_call = MockToolCall('{"query": "test"}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert parsed["limit"] == 10
    assert parsed["offset"] == 0


def test_tool_handler_factory_argument_parsing_error():
    """Test that argument parsing errors include tool name."""

    def simple_tool(query: str):
        return {"query": query}

    handler = tool_handler_factory(simple_tool)

    # Invalid JSON
    tool_call = MockToolCall('invalid json', name="my_tool")

    with pytest.raises(ValueError) as exc_info:
        handler(tool_call)

    error_msg = str(exc_info.value)
    assert "my_tool" in error_msg


def test_tool_handler_factory_returns_string():
    """Test that the result from handler is a JSON string."""

    def tool_returning_list():
        return ["a", "b", "c"]

    handler = tool_handler_factory(tool_returning_list)

    tool_call = MockToolCall('{}')
    result = handler(tool_call)

    # Should be a string
    assert isinstance(result, str)

    # Should be valid JSON
    parsed = json.loads(result)
    assert parsed == ["a", "b", "c"]

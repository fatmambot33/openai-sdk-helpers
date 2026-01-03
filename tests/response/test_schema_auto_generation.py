"""Tests for automatic schema generation from output_structure."""

import pytest
from pydantic import Field
from unittest.mock import Mock, patch

from openai_sdk_helpers.config import OpenAISettings
from openai_sdk_helpers.response.base import BaseResponse
from openai_sdk_helpers.response.config import ResponseConfiguration
from openai_sdk_helpers.structure.base import BaseStructure


class DummyOutputStructure(BaseStructure):
    """Test structure for schema auto-generation."""

    text: str = Field(description="Test text field")
    score: float = Field(description="Test score field")


def test_schema_auto_generated_from_output_structure(openai_settings):
    """Test that schema is auto-generated from output_structure."""
    instance = BaseResponse(
        instructions="Test instructions",
        tools=None,
        output_structure=DummyOutputStructure,
        tool_handlers={},
        openai_settings=openai_settings,
    )

    # output_structure should be stored
    assert instance._output_structure is not None
    assert instance._output_structure == DummyOutputStructure
    # response_format() should return a dict with a 'format' key
    schema = instance._output_structure.response_format()
    assert isinstance(schema, dict)
    assert "format" in schema


def test_schema_auto_generated_even_with_tools(openai_settings):
    """Test that output_structure is stored even when tools are present."""
    instance = BaseResponse(
        instructions="Test instructions",
        tools=[{"type": "function", "name": "test_tool"}],
        output_structure=DummyOutputStructure,
        tool_handlers={},
        openai_settings=openai_settings,
    )

    # output_structure should be stored even with tools
    assert instance._output_structure is not None
    assert instance._output_structure == DummyOutputStructure
    # response_format() should return a dict with a 'format' key
    schema = instance._output_structure.response_format()
    assert isinstance(schema, dict)
    assert "format" in schema


def test_schema_none_when_no_output_structure(openai_settings):
    """Test that output_structure is None when not provided."""
    instance = BaseResponse(
        instructions="Test instructions",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )

    # output_structure should be None
    assert instance._output_structure is None


def test_response_configuration_auto_generates_schema():
    """Test that ResponseConfiguration stores output_structure in gen_response."""
    config = ResponseConfiguration(
        name="test_config",
        instructions="Test instructions",
        tools=None,
        input_structure=None,
        output_structure=DummyOutputStructure,
    )

    # But when generating a response, it should store the output_structure
    settings = OpenAISettings(api_key="test-key", default_model="gpt-4o-mini")

    response = config.gen_response(openai_settings=settings)

    # The generated response should have the output_structure
    assert response._output_structure is not None
    assert response._output_structure == DummyOutputStructure


def test_schema_used_only_when_no_tools(openai_settings):
    """Test that schema is only sent to API when no tools are present."""
    # Case 1: No tools, with output_structure -> schema should be used
    instance = BaseResponse(
        instructions="Test instructions",
        tools=None,
        output_structure=DummyOutputStructure,
        tool_handlers={},
        openai_settings=openai_settings,
    )

    with patch.object(instance._client.responses, "create") as mock_create:
        mock_create.return_value = Mock(output=[])
        try:
            instance.run_sync("Test content")
        except RuntimeError:
            pass  # Expected: "No output returned from OpenAI."

        call_kwargs = mock_create.call_args[1]
        assert "text" in call_kwargs
        assert "tools" not in call_kwargs

    # Case 2: With tools, with output_structure -> schema should NOT be used
    instance_with_tools = BaseResponse(
        instructions="Test instructions",
        tools=[{"type": "function", "name": "test_tool"}],
        output_structure=DummyOutputStructure,
        tool_handlers={"test_tool": lambda x: "{}"},
        openai_settings=openai_settings,
    )

    with patch.object(instance_with_tools._client.responses, "create") as mock_create:
        mock_create.return_value = Mock(output=[])
        try:
            instance_with_tools.run_sync("Test content")
        except RuntimeError:
            pass  # Expected: "No output returned from OpenAI."

        call_kwargs = mock_create.call_args[1]
        assert "text" not in call_kwargs
        assert "tools" in call_kwargs

"""Tests for automatic schema generation from output_structure."""

import pytest
from pydantic import Field

from openai_sdk_helpers.response.base import BaseResponse
from openai_sdk_helpers.response.config import ResponseConfiguration
from openai_sdk_helpers.structure.base import BaseStructure


class DummyOutputStructure(BaseStructure):
    """Test structure for schema auto-generation."""

    text: str = Field(description="Test text field")
    score: float = Field(description="Test score field")


def test_schema_auto_generated_when_none_and_no_tools(openai_settings):
    """Test that schema is auto-generated from output_structure when schema is None and no tools."""
    # When schema is None and no tools, it should be auto-generated from output_structure
    instance = BaseResponse(
        instructions="Test instructions",
        tools=None,
        schema=None,
        output_structure=DummyOutputStructure,
        tool_handlers={},
        openai_settings=openai_settings,
    )

    # Schema should have been auto-generated
    assert instance._schema is not None
    # It should be a dict with a 'format' key (ResponseTextConfigParam structure)
    assert isinstance(instance._schema, dict)
    assert "format" in instance._schema


def test_schema_not_auto_generated_when_tools_present(openai_settings):
    """Test that schema is NOT auto-generated when tools are present."""
    # When tools are present, schema should NOT be auto-generated
    instance = BaseResponse(
        instructions="Test instructions",
        tools=[{"type": "function", "name": "test_tool"}],
        schema=None,
        output_structure=DummyOutputStructure,
        tool_handlers={},
        openai_settings=openai_settings,
    )

    # Schema should remain None when tools are present
    assert instance._schema is None


def test_schema_not_auto_generated_when_explicitly_provided(openai_settings):
    """Test that explicit schema is not overridden."""
    explicit_schema = DummyOutputStructure.response_format()

    instance = BaseResponse(
        instructions="Test instructions",
        tools=None,
        schema=explicit_schema,
        output_structure=DummyOutputStructure,
        tool_handlers={},
        openai_settings=openai_settings,
    )

    # Schema should be the explicitly provided one
    assert instance._schema is explicit_schema


def test_schema_not_auto_generated_when_no_output_structure(openai_settings):
    """Test that schema is not auto-generated when output_structure is None."""
    instance = BaseResponse(
        instructions="Test instructions",
        tools=None,
        schema=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )

    # Schema should remain None
    assert instance._schema is None


def test_response_configuration_auto_generates_schema():
    """Test that ResponseConfiguration auto-generates schema in gen_response."""
    config = ResponseConfiguration(
        name="test_config",
        instructions="Test instructions",
        tools=None,
        schema=None,
        input_structure=None,
        output_structure=DummyOutputStructure,
    )

    # Schema should be None in config itself (not auto-generated yet)
    assert config.schema is None

    # But when generating a response, it should auto-generate
    from openai_sdk_helpers.config import OpenAISettings

    settings = OpenAISettings(api_key="test-key", default_model="gpt-4o-mini")

    response = config.gen_response(openai_settings=settings)

    # The generated response should have auto-generated schema
    assert response._schema is not None

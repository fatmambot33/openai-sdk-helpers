"""Tests for ResponseConfiguration instruction handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import Field

from openai_sdk_helpers.response.configuration import ResponseConfiguration
from openai_sdk_helpers.structure.base import StructureBase


def _build_config(instructions: str | Path) -> ResponseConfiguration:
    return ResponseConfiguration(
        name="unit",
        instructions=instructions,
        tools=None,
        input_structure=None,
        output_structure=None,
    )


def test_instructions_text_returns_plain_string() -> None:
    configuration = _build_config("Use direct instructions.")
    assert configuration.get_resolved_instructions == "Use direct instructions."


def test_instructions_text_reads_template_file(tmp_path: Path) -> None:
    template_path = tmp_path / "template.jinja"
    template_path.write_text("Template instructions", encoding="utf-8")

    configuration = _build_config(template_path)
    assert configuration.get_resolved_instructions == "Template instructions"


def test_empty_string_instructions_raise_value_error() -> None:
    with pytest.raises(ValueError):
        _build_config("   ")


def test_missing_template_raises_file_not_found(tmp_path: Path) -> None:
    missing_template = tmp_path / "missing.jinja"
    with pytest.raises(FileNotFoundError):
        _build_config(missing_template)


def test_invalid_instruction_type_raises_type_error() -> None:
    invalid_instructions = cast(Any, 123)
    with pytest.raises(TypeError):
        ResponseConfiguration(
            name="unit",
            instructions=invalid_instructions,
            tools=None,
            input_structure=None,
            output_structure=None,
        )


class _SampleOutput(StructureBase):
    """Sample output structure for instruction generation tests."""

    summary: str = Field(description="Brief summary of the content")


def test_output_instructions_are_appended(openai_settings) -> None:
    configuration = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=_SampleOutput,
    )

    response = configuration.gen_response(openai_settings=openai_settings)

    expected_instructions = configuration.get_resolved_instructions
    assert response._instructions == expected_instructions


def test_output_instructions_can_be_skipped(openai_settings) -> None:
    configuration = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=_SampleOutput,
        add_output_instructions=False,
    )

    response = configuration.gen_response(openai_settings=openai_settings)

    expected_instructions = configuration.get_resolved_instructions

    assert response._instructions == expected_instructions


def test_no_output_structure_ignores_add_output_instructions(
    openai_settings,
) -> None:
    """Test that when output_structure is None, add_output_instructions has no effect."""
    config_true = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        add_output_instructions=True,
    )
    config_false = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        add_output_instructions=False,
    )

    response_with_flag = config_true.gen_response(openai_settings=openai_settings)
    response_without_flag = config_false.gen_response(openai_settings=openai_settings)

    # Both should produce the same result: just the base instructions
    assert response_with_flag._instructions == config_true.get_resolved_instructions
    assert response_without_flag._instructions == config_false.get_resolved_instructions


def test_invalid_string_tools_container_raises_type_error() -> None:
    """Reject string tool containers during configuration initialization."""
    with pytest.raises(TypeError, match="non-string sequence"):
        ResponseConfiguration(
            name="unit",
            instructions="Base instructions",
            tools=cast(Any, "abc"),
            input_structure=None,
            output_structure=None,
        )


def test_invalid_bytes_tools_container_raises_type_error() -> None:
    """Reject byte-string tool containers during configuration initialization."""
    with pytest.raises(TypeError, match="non-string sequence"):
        ResponseConfiguration(
            name="unit",
            instructions="Base instructions",
            tools=cast(Any, b"abc"),
            input_structure=None,
            output_structure=None,
        )


def test_invalid_tool_item_type_raises_type_error() -> None:
    """Reject non-mapping tool definitions during configuration initialization."""
    with pytest.raises(TypeError, match="items must be mappings"):
        ResponseConfiguration(
            name="unit",
            instructions="Base instructions",
            tools=cast(Any, [123]),
            input_structure=None,
            output_structure=None,
        )


def test_add_web_search_tool_runs_end_to_end(
    openai_settings, mock_openai_client
) -> None:
    """Include the web_search tool definition when configured."""
    mock_openai_client.responses.create.return_value = cast(
        Any,
        type("Response", (), {"output": [object()], "output_text": "ok"})(),
    )

    configuration = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        add_web_search_tool=True,
    )

    response = configuration.gen_response(openai_settings=openai_settings)
    result = response.run_sync("hello")

    assert result == "ok"
    create_kwargs = mock_openai_client.responses.create.call_args.kwargs
    assert create_kwargs["tools"] == [{"type": "web_search"}]
    assert create_kwargs["tool_choice"] == "auto"


def test_save_messages_is_applied_to_response(openai_settings) -> None:
    """Test that save_messages defaults are passed to ResponseBase."""
    configuration = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        save_messages=False,
    )

    response = configuration.gen_response(openai_settings=openai_settings)

    assert response._save_messages is False

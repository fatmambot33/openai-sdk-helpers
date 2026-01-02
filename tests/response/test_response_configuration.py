"""Tests for ResponseConfiguration instruction handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from openai_sdk_helpers.response.config import ResponseConfiguration


def _build_config(instructions: str | Path) -> ResponseConfiguration:
    return ResponseConfiguration(
        name="unit",
        instructions=instructions,
        tools=None,
        schema=None,
        input_structure=None,
        output_structure=None,
    )


def test_instructions_text_returns_plain_string() -> None:
    config = _build_config("Use direct instructions.")
    assert config.instructions_text == "Use direct instructions."


def test_instructions_text_reads_template_file(tmp_path: Path) -> None:
    template_path = tmp_path / "template.jinja"
    template_path.write_text("Template instructions", encoding="utf-8")

    config = _build_config(template_path)
    assert config.instructions_text == "Template instructions"


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
            schema=None,
            input_structure=None,
            output_structure=None,
        )

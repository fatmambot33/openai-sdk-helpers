"""Tests for extractor prompt and configuration generators."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from openai_sdk_helpers.extract import generator
from openai_sdk_helpers.settings import OpenAISettings
from openai_sdk_helpers.structure.extraction import (
    DocumentExtractorConfig,
    ExampleData,
    Extraction,
)
from openai_sdk_helpers.structure.prompt import PromptStructure


def test_optimize_extractor_prompt_uses_prompter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure prompt optimization uses the prompter response configuration."""
    openai_settings = OpenAISettings(api_key="test", default_model="gpt-4o-mini")
    mock_response = Mock()
    mock_response.run_sync.return_value = PromptStructure(prompt="optimized prompt")

    monkeypatch.setattr(
        generator,
        "PROMPTER",
        Mock(gen_response=Mock(return_value=mock_response)),
    )

    result = generator.optimize_extractor_prompt(
        openai_settings,
        "Extract names and dates.",
        ["Name", "Date"],
        additional_context="Focus on invoices.",
    )

    assert result == "optimized prompt"
    mock_response.run_sync.assert_called_once()
    request_text = mock_response.run_sync.call_args[0][0]
    assert "Extract names and dates." in request_text
    assert "- Name" in request_text
    assert "- Date" in request_text
    assert "Focus on invoices." in request_text
    mock_response.close.assert_called_once()


def test_generate_document_extractor_config_uses_generator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ensure config generation uses the response-based generator."""
    openai_settings = OpenAISettings(api_key="test", default_model="gpt-4o-mini")
    source_file = tmp_path.mktemp("examples") / "invoice.txt"
    source_file.write_text("Invoice ACME-001 lists Widget A for $10.")
    examples = [
        ExampleData(
            text="ROMEO. But soft!",
            extractions=[
                Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"emotional_state": "wonder"},
                )
            ],
        )
    ]
    expected = DocumentExtractorConfig(
        name="character_extractor",
        prompt_description="optimized prompt",
        extraction_classes=["Name"],
        examples=examples,
    )

    monkeypatch.setattr(
        generator, "optimize_extractor_prompt", Mock(return_value="optimized prompt")
    )
    mock_response = Mock()
    mock_response.run_sync.return_value = expected
    monkeypatch.setattr(
        generator,
        "EXTRACTOR_CONFIG_GENERATOR",
        Mock(gen_response=Mock(return_value=mock_response)),
    )

    result = generator.generate_document_extractor_config(
        openai_settings,
        "character_extractor",
        "Extract names.",
        ["Name"],
        example_files=[source_file],
    )

    assert result == expected
    generator.optimize_extractor_prompt.assert_called_once_with(
        openai_settings,
        "Extract names.",
        ["Name"],
        additional_context=None,
    )
    mock_response.run_sync.assert_called_once()
    request_text = mock_response.run_sync.call_args[0][0]
    assert "Name: character_extractor" in request_text
    assert "Prompt description: optimized prompt" in request_text
    assert "- Name" in request_text
    assert "Example requirements:" in request_text
    assert "Generate 3 high-quality examples" in request_text
    assert "Attributes guidance:" in request_text
    assert str(source_file) in request_text
    assert "Invoice ACME-001 lists Widget A for $10." in request_text
    mock_response.close.assert_called_once()


def test_optimize_extractor_prompt_with_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure agent-driven prompt optimization uses AgentBase."""
    openai_settings = OpenAISettings(api_key="test", default_model="gpt-4o-mini")
    captured: dict[str, object] = {}

    class DummyAgent:
        def __init__(self, configuration: object) -> None:
            captured["configuration"] = configuration

        def run_sync(self, input: str) -> PromptStructure:
            captured["input"] = input
            return PromptStructure(prompt="agent-optimized")

    monkeypatch.setattr(generator, "AgentBase", DummyAgent)

    result = generator.optimize_extractor_prompt_with_agent(
        openai_settings,
        "Extract vendors.",
        ["Vendor"],
        additional_context="Invoices only.",
    )

    assert result == "agent-optimized"
    configuration = captured["configuration"]
    assert isinstance(configuration, generator.AgentConfiguration)
    assert configuration.model == openai_settings.default_model
    assert "Extract vendors." in captured["input"]


def test_generate_document_extractor_config_with_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure agent-driven config generation uses AgentBase."""
    openai_settings = OpenAISettings(api_key="test", default_model="gpt-4o-mini")
    examples = [
        ExampleData(
            text="Juliet is the sun.",
            extractions=[
                Extraction(
                    extraction_class="relationship",
                    extraction_text="Juliet is the sun",
                    attributes={"type": "metaphor"},
                )
            ],
        )
    ]
    expected = DocumentExtractorConfig(
        name="vendor_extractor",
        prompt_description="agent prompt",
        extraction_classes=["Vendor"],
        examples=examples,
    )
    captured: dict[str, object] = {}

    class DummyAgent:
        def __init__(self, configuration: object) -> None:
            captured["configuration"] = configuration

        def run_sync(self, input: str) -> DocumentExtractorConfig:
            captured["input"] = input
            return expected

    monkeypatch.setattr(
        generator,
        "optimize_extractor_prompt_with_agent",
        Mock(return_value="agent prompt"),
    )
    monkeypatch.setattr(generator, "AgentBase", DummyAgent)

    result = generator.generate_document_extractor_config_with_agent(
        openai_settings,
        "vendor_extractor",
        "Extract vendors.",
        ["Vendor"],
        examples,
    )

    assert result == expected
    configuration = captured["configuration"]
    assert isinstance(configuration, generator.AgentConfiguration)
    assert configuration.model == openai_settings.default_model
    assert "Name: vendor_extractor" in captured["input"]
    assert "Prompt description: agent prompt" in captured["input"]

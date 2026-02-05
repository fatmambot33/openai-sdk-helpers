from __future__ import annotations

from unittest.mock import patch

from openai_sdk_helpers.agent.extractor import ExtractorAgent
from openai_sdk_helpers.structure.extraction import (
    DocumentExtractorConfig,
    ExampleDataStructure,
)


def test_extractor_agent_from_config_builds_extractor():
    """Ensure ExtractorAgent uses configuration values."""

    config = DocumentExtractorConfig(
        name="example",
        prompt_description="Extract entities.",
        extraction_classes=["entity"],
        examples=[ExampleDataStructure(text="Example text")],
    )

    with patch(
        "openai_sdk_helpers.agent.extractor.DocumentExtractor"
    ) as mock_extractor:
        agent = ExtractorAgent.from_config(config, model="gpt-4o-mini", max_workers=2)

    mock_extractor.assert_called_once_with(
        prompt_description="Extract entities.",
        examples=config.examples,
        model_id="gpt-4o-mini",
        max_workers=2,
    )
    assert isinstance(agent, ExtractorAgent)


def test_extractor_agent_extract_text_builds_documents():
    """Ensure ExtractorAgent wraps raw text into document structures."""

    with patch(
        "openai_sdk_helpers.agent.extractor.DocumentExtractor"
    ) as mock_extractor:
        mock_extractor.return_value.extract.return_value = ["result"]
        agent = ExtractorAgent(
            prompt_description="Extract entities.",
            examples=[ExampleDataStructure(text="Example text")],
            model="gpt-4o-mini",
        )
        result = agent.extract_text(
            ["First document", "Second document"], additional_context="context"
        )

    assert result == ["result"]
    documents = mock_extractor.return_value.extract.call_args.args[0]
    assert len(documents) == 2
    assert documents[0].text == "First document"
    assert documents[0].additional_context == "context"
    assert documents[1].text == "Second document"
    assert documents[1].additional_context == "context"


def test_extractor_agent_renders_prompt_template_with_env(tmp_path, monkeypatch):
    """Ensure ExtractorAgent renders a Jinja template with env variables."""

    monkeypatch.setenv("EXTRACTOR_PROMPT", "Extract entities")
    template_path = tmp_path / "prompt.jinja"
    template_path.write_text("{{ env.EXTRACTOR_PROMPT }}: {{ suffix }}")

    with patch(
        "openai_sdk_helpers.agent.extractor.DocumentExtractor"
    ) as mock_extractor:
        ExtractorAgent(
            prompt_description=None,
            examples=[ExampleDataStructure(text="Example text")],
            model="gpt-4o-mini",
            template_path=template_path,
            template_context={"suffix": "from template"},
        )

    mock_extractor.assert_called_once_with(
        prompt_description="Extract entities: from template",
        examples=[ExampleDataStructure(text="Example text")],
        model_id="gpt-4o-mini",
        max_workers=1,
    )

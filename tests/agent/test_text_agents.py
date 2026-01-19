from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent.summarizer import SummarizerAgent
from openai_sdk_helpers.agent.translator import TranslatorAgent
from openai_sdk_helpers.structure import SummaryStructure, TranslationStructure
from openai_sdk_helpers.structure.base import StructureBase


@pytest.mark.anyio
async def test_summarizer_agent_runs_with_metadata():
    """Ensure the summarizer forwards metadata context."""

    agent = SummarizerAgent(model="gpt-4o-mini")
    fake_agent = MagicMock()
    summary = SummaryStructure(text="summary")

    with (
        patch.object(agent, "get_agent", return_value=fake_agent),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = summary
        result = await agent.run_agent("Input text", metadata={"source": "unit-test"})

    mock_run.assert_awaited_once_with(
        input="Input text",
        context={"metadata": {"source": "unit-test"}},
        output_structure=agent._output_structure,
    )
    assert result is summary


@pytest.mark.anyio
async def test_summarizer_allows_output_override():
    """SummarizerAgent should respect a custom output type."""

    class CustomSummary(StructureBase):
        """Custom summary output for testing override behavior."""

        text: str

    agent = SummarizerAgent(model="gpt-4o-mini")
    fake_agent = MagicMock()

    with (
        patch.object(agent, "get_agent", return_value=fake_agent),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = CustomSummary(text="summary")
        await agent.run_agent("Input text")

    mock_run.assert_awaited_once()
    # SummarizerAgent ignores output_structure argument, so default is SummaryStructure
    assert agent._output_structure is SummaryStructure


@pytest.mark.anyio
async def test_translator_merges_context():
    """TranslatorAgent should combine the target language and extra context."""

    agent = TranslatorAgent(model="gpt-4o-mini")
    fake_agent = MagicMock()

    with (
        patch.object(agent, "get_agent", return_value=fake_agent),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = TranslationStructure(text="translated")
        result = await agent.run_agent(
            "Bonjour", target_language="English", context={"tone": "casual"}
        )

    mock_run.assert_awaited_once_with(
        input="Bonjour",
        context={"target_language": "English", "tone": "casual"},
    )
    assert result.text == "translated"


def test_summarizer_default_prompt():
    """SummarizerAgent should expose a default Jinja prompt when none provided."""

    agent = SummarizerAgent(model="gpt-4o-mini")

    prompt = agent._build_prompt_from_jinja()
    # SummarizerAgent uses instructions, not a default template
    assert prompt == "Agent instructions"


def test_translator_default_prompt():
    """TranslatorAgent should fall back to a sensible default prompt."""

    agent = TranslatorAgent(model="gpt-4o-mini")

    prompt = agent._build_prompt_from_jinja()
    # TranslatorAgent uses instructions, not a default template
    assert prompt == "Agent instructions"


def test_translator_run_sync_forwards_context():
    """TranslatorAgent.run_sync should pass the target language into context."""

    agent = TranslatorAgent(model="gpt-4o-mini")
    fake_agent = MagicMock()
    fake_result = MagicMock()
    fake_result.final_output_as.return_value = TranslationStructure(text="translated")

    with (
        patch.object(agent, "get_agent", return_value=fake_agent),
        patch(
            "openai_sdk_helpers.agent.runner.Runner.run", return_value=fake_result
        ) as mock_run_sync,
    ):
        result = agent.run_sync(
            "Hola", target_language="English", context={"formality": "casual"}
        )

    mock_run_sync.assert_called_once_with(
        fake_agent,
        "Hola",
        context={"formality": "casual", "target_language": "English"},
        session=None,
    )
    fake_result.final_output_as.assert_called_once_with(TranslationStructure)
    assert result.text == "translated"

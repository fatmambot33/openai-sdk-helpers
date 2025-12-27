from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent.summarizer import SummarizerAgent
from openai_sdk_helpers.agent.translator import TranslatorAgent
from openai_sdk_helpers.structure import SummaryStructure


@pytest.mark.anyio
async def test_summarizer_agent_runs_with_metadata():
    """Ensure the summarizer forwards metadata context."""

    agent = SummarizerAgent(default_model="gpt-4o-mini")
    fake_agent = MagicMock()
    summary = SummaryStructure(text="summary")

    with (
        patch.object(agent, "get_agent", return_value=fake_agent),
        patch(
            "openai_sdk_helpers.agent.summarizer._run_agent", new_callable=AsyncMock
        ) as mock_run,
    ):
        mock_run.return_value = summary
        result = await agent.run_agent("Input text", metadata={"source": "unit-test"})

    mock_run.assert_awaited_once_with(
        agent=fake_agent,
        agent_input="Input text",
        agent_context={"metadata": {"source": "unit-test"}},
        output_type=agent._output_type,
    )
    assert result is summary


@pytest.mark.anyio
async def test_summarizer_allows_output_override():
    """SummarizerAgent should respect a custom output type."""

    agent = SummarizerAgent(default_model="gpt-4o-mini", output_type=str)
    fake_agent = MagicMock()

    with (
        patch.object(agent, "get_agent", return_value=fake_agent),
        patch(
            "openai_sdk_helpers.agent.summarizer._run_agent", new_callable=AsyncMock
        ) as mock_run,
    ):
        mock_run.return_value = "summary"
        await agent.run_agent("Input text")

    mock_run.assert_awaited_once()
    assert agent._output_type is str


@pytest.mark.anyio
async def test_translator_merges_context():
    """TranslatorAgent should combine the target language and extra context."""

    agent = TranslatorAgent(default_model="gpt-4o-mini")
    fake_agent = MagicMock()

    with (
        patch.object(agent, "get_agent", return_value=fake_agent),
        patch(
            "openai_sdk_helpers.agent.translator._run_agent", new_callable=AsyncMock
        ) as mock_run,
    ):
        mock_run.return_value = "translated"
        result = await agent.run_agent(
            "Bonjour", target_language="English", context={"tone": "casual"}
        )

    mock_run.assert_awaited_once_with(
        agent=fake_agent,
        agent_input="Bonjour",
        agent_context={"target_language": "English", "tone": "casual"},
        output_type=str,
    )
    assert result == "translated"


def test_summarizer_default_prompt():
    """SummarizerAgent should expose a default Jinja prompt when none provided."""

    agent = SummarizerAgent(default_model="gpt-4o-mini")

    prompt = agent._build_instructions_from_jinja()

    assert "summarizes long-form text" in prompt
    assert "bullet points" in prompt


def test_translator_default_prompt():
    """TranslatorAgent should fall back to a sensible default prompt."""

    agent = TranslatorAgent(default_model="gpt-4o-mini")

    prompt = agent._build_instructions_from_jinja()

    assert "professional translator" in prompt
    assert "target language" in prompt


def test_translator_run_sync_forwards_context():
    """TranslatorAgent.run_sync should pass the target language into context."""

    agent = TranslatorAgent(default_model="gpt-4o-mini")
    fake_agent = MagicMock()
    fake_result = MagicMock()
    fake_result.final_output_as.return_value = "translated"

    with (
        patch.object(agent, "get_agent", return_value=fake_agent),
        patch(
            "openai_sdk_helpers.agent.base._run_agent_sync", return_value=fake_result
        ) as mock_run_sync,
    ):
        result = agent.run_sync(
            "Hola", target_language="English", context={"formality": "casual"}
        )

    mock_run_sync.assert_called_once_with(
        fake_agent,
        "Hola",
        agent_context={"target_language": "English", "formality": "casual"},
    )
    fake_result.final_output_as.assert_called_once_with(str)
    assert result == "translated"

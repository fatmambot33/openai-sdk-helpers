from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent.validation import ValidatorAgent
from openai_sdk_helpers.structure import ValidationResultStructure


@pytest.mark.anyio
async def test_validator_agent_merges_context():
    """ValidatorAgent should merge optional context into the guardrail check."""

    agent = ValidatorAgent(default_model="gpt-4o-mini")
    fake_agent = MagicMock()
    validation = ValidationResultStructure(
        input_safe=True,
        output_safe=True,
        violations=[],
        recommended_actions=[],
        sanitized_output=None,
    )

    with patch.object(agent, "get_agent", return_value=fake_agent), patch(
        "openai_sdk_helpers.agent.validation._run_agent", new_callable=AsyncMock
    ) as mock_run:
        mock_run.return_value = validation
        result = await agent.run_agent(
            "User asks for advice",
            agent_output="Agent response",
            policy_notes="No PII",
            extra_context={"session_id": "abc123"},
        )

    mock_run.assert_awaited_once_with(
        agent=fake_agent,
        agent_input="User asks for advice",
        agent_context={
            "user_input": "User asks for advice",
            "agent_output": "Agent response",
            "policy_notes": "No PII",
            "session_id": "abc123",
        },
        output_type=ValidationResultStructure,
    )
    assert result is validation


def test_validator_agent_default_prompt():
    """ValidatorAgent should provide a guardrail-focused default prompt."""

    agent = ValidatorAgent(default_model="gpt-4o-mini")

    prompt = agent._build_instructions_from_jinja()

    assert "safety validator" in prompt
    assert "guardrails" in prompt
    assert "user_input" in prompt

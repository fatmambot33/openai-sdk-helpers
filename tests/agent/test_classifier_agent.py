from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent.classifier import TaxonomyClassifierAgent
from openai_sdk_helpers.structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    TaxonomyNode,
)


def test_classifier_default_prompt_template():
    """Classifier should use the bundled classifier prompt by default."""

    agent = TaxonomyClassifierAgent(model="gpt-4o-mini")

    prompt = agent._build_prompt_from_jinja()
    assert "taxonomy classification assistant" in prompt


@pytest.mark.anyio
async def test_classifier_traverses_taxonomy_levels():
    """Classifier should walk the taxonomy until a terminal step."""

    agent = TaxonomyClassifierAgent(model="gpt-4o-mini")
    root = TaxonomyNode(
        id="finance",
        label="Finance",
        children=[TaxonomyNode(id="tax", label="Tax")],
    )
    alternate = TaxonomyNode(id="health", label="Health")
    steps = [
        ClassificationStep(
            selected_id="finance",
            selected_label="Finance",
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="tax",
            selected_label="Tax",
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_agent("Tax update", taxonomy=[root, alternate])

    assert isinstance(result, ClassificationResult)
    assert result.final_id == "tax"
    assert result.final_label == "Tax"
    assert result.stop_reason is ClassificationStopReason.STOP
    assert len(result.path) == 2


@pytest.mark.anyio
async def test_classifier_stops_when_no_children():
    """Classifier should stop when a selected node has no children."""

    agent = TaxonomyClassifierAgent(model="gpt-4o-mini")
    root = TaxonomyNode(id="root", label="Root")
    step = ClassificationStep(
        selected_id="root",
        selected_label="Root",
        confidence=0.6,
        stop_reason=ClassificationStopReason.CONTINUE,
    )

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = step
        result = await agent.run_agent("Root only", taxonomy=[root])

    assert result.stop_reason is ClassificationStopReason.NO_CHILDREN
    assert result.final_id == "root"


@pytest.mark.anyio
async def test_classifier_requires_taxonomy_nodes():
    """Classifier should reject empty taxonomy definitions."""

    agent = TaxonomyClassifierAgent(model="gpt-4o-mini")

    with pytest.raises(ValueError, match="taxonomy must include at least one node"):
        await agent.run_agent("Text", taxonomy=[])

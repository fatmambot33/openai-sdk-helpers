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

    agent = TaxonomyClassifierAgent(
        model="gpt-4o-mini", taxonomy=TaxonomyNode(id="root", label="Root")
    )

    prompt = agent._build_prompt_from_jinja()
    assert "taxonomy classification assistant" in prompt


@pytest.mark.anyio
async def test_classifier_traverses_taxonomy_levels():
    """Classifier should walk the taxonomy until a terminal step."""
    root = TaxonomyNode(
        id="finance",
        label="Finance",
        children=[TaxonomyNode(id="tax", label="Tax")],
    )
    alternate = TaxonomyNode(id="health", label="Health")
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root, alternate])

    steps = [
        ClassificationStep(
            selected_id="finance",
            selected_ids=["finance"],
            selected_label="Finance",
            selected_labels=["Finance"],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="tax",
            selected_ids=["tax"],
            selected_label="Tax",
            selected_labels=["Tax"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_agent("Tax update")

    assert isinstance(result, ClassificationResult)
    assert result.final_node is not None
    assert result.final_node.id == "tax"
    assert result.final_nodes is not None
    assert [node.id for node in result.final_nodes] == ["tax"]
    assert [node.id for node in result.path_nodes] == ["finance", "tax"]
    assert result.stop_reason is ClassificationStopReason.STOP
    assert len(result.path) == 2


@pytest.mark.anyio
async def test_classifier_traverses_multiple_branches():
    """Classifier should recurse into multiple selected branches."""
    meat = TaxonomyNode(
        id="meat",
        label="Meat",
        children=[TaxonomyNode(id="beef", label="Beef")],
    )
    vegetables = TaxonomyNode(
        id="vegetables",
        label="Vegetables",
        children=[TaxonomyNode(id="carrot", label="Carrot")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[meat, vegetables])

    steps = [
        ClassificationStep(
            selected_ids=["meat", "vegetables"],
            selected_labels=["Meat", "Vegetables"],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_ids=["beef"],
            selected_labels=["Beef"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
        ClassificationStep(
            selected_ids=["carrot"],
            selected_labels=["Carrot"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_agent("Culinary update")

    assert result.final_nodes is not None
    assert [node.id for node in result.final_nodes] == ["beef", "carrot"]
    assert [node.id for node in result.path_nodes] == [
        "meat",
        "vegetables",
        "beef",
        "carrot",
    ]
    assert result.path[-1].selected_ids == ["carrot"]


@pytest.mark.anyio
async def test_classifier_single_class_limits_branches():
    """Classifier should limit traversal to a single branch when enabled."""
    meat = TaxonomyNode(
        id="meat",
        label="Meat",
        children=[TaxonomyNode(id="beef", label="Beef")],
    )
    vegetables = TaxonomyNode(
        id="vegetables",
        label="Vegetables",
        children=[TaxonomyNode(id="carrot", label="Carrot")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[meat, vegetables])

    steps = [
        ClassificationStep(
            selected_ids=["meat", "vegetables"],
            selected_labels=["Meat", "Vegetables"],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_ids=["beef"],
            selected_labels=["Beef"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_agent("Culinary update", single_class=True)

    assert result.final_nodes is not None
    assert [node.id for node in result.final_nodes] == ["beef"]
    assert [node.id for node in result.path_nodes] == ["meat", "beef"]


@pytest.mark.anyio
async def test_classifier_confidence_threshold_stops_branch():
    """Classifier should stop a branch when confidence is below the threshold."""
    root = TaxonomyNode(
        id="root",
        label="Root",
        children=[TaxonomyNode(id="child", label="Child")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    step = ClassificationStep(
        selected_ids=["root"],
        selected_labels=["Root"],
        confidence=0.2,
        stop_reason=ClassificationStopReason.CONTINUE,
    )

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = step
        result = await agent.run_agent("Low confidence", confidence_threshold=0.5)

    assert result.stop_reason is ClassificationStopReason.NO_MATCH
    assert result.final_nodes is None


@pytest.mark.anyio
async def test_classifier_stops_when_no_children():
    """Classifier should stop when a selected node has no children."""

    root = TaxonomyNode(id="root", label="Root")
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    step = ClassificationStep(
        selected_id="root",
        selected_ids=["root"],
        selected_label="Root",
        selected_labels=["Root"],
        confidence=0.6,
        stop_reason=ClassificationStopReason.CONTINUE,
    )

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = step
        result = await agent.run_agent("Root only")

    assert result.stop_reason is ClassificationStopReason.NO_CHILDREN
    assert result.final_node is not None
    assert result.final_node.id == "root"


@pytest.mark.anyio
async def test_classifier_falls_back_when_selected_ids_empty():
    """Classifier should fall back to selected_id when selected_ids is empty."""

    root = TaxonomyNode(
        id="finance",
        label="Finance",
        children=[TaxonomyNode(id="tax", label="Tax")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    steps = [
        ClassificationStep(
            selected_id="finance",
            selected_ids=[""],
            selected_label="Finance",
            selected_labels=None,
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="tax",
            selected_ids=["tax"],
            selected_label="Tax",
            selected_labels=["Tax"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_agent("Tax update")

    assert result.stop_reason is ClassificationStopReason.STOP
    assert result.final_node is not None
    assert result.final_node.id == "tax"


@pytest.mark.anyio
async def test_classifier_falls_back_when_selected_labels_empty():
    """Classifier should fall back to selected_label when selected_labels is empty."""

    root = TaxonomyNode(
        id="finance",
        label="Finance",
        children=[TaxonomyNode(id="tax", label="Tax")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    steps = [
        ClassificationStep(
            selected_id=None,
            selected_ids=None,
            selected_label="Finance",
            selected_labels=[""],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="tax",
            selected_ids=["tax"],
            selected_label="Tax",
            selected_labels=["Tax"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_agent("Tax update")

    assert result.stop_reason is ClassificationStopReason.STOP
    assert result.final_node is not None
    assert result.final_node.label == "Tax"


@pytest.mark.anyio
async def test_classifier_requires_taxonomy_nodes():
    """Classifier should reject empty taxonomy definitions."""

    with pytest.raises(ValueError, match="taxonomy must include at least one node"):
        TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[])

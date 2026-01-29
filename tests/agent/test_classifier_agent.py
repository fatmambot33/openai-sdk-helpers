from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent.classifier import (
    TaxonomyClassifierAgent,
    _build_node_path_map,
)
from openai_sdk_helpers.structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    TaxonomyNode,
)


def test_classifier_default_prompt_template():
    """Classifier should use the bundled classifier prompt by default."""

    agent = TaxonomyClassifierAgent(
        model="gpt-4o-mini", taxonomy=TaxonomyNode(label="Root")
    )

    prompt = agent._build_prompt_from_jinja()
    assert "taxonomy classification assistant" in prompt


@pytest.mark.anyio
async def test_classifier_traverses_taxonomy_levels():
    """Classifier should walk the taxonomy until a terminal step."""
    root = TaxonomyNode(
        label="Finance",
        children=[TaxonomyNode(label="Tax")],
    )
    alternate = TaxonomyNode(label="Health")
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root, alternate])

    steps = [
        ClassificationStep(
            selected_id="Finance",
            selected_ids=["Finance"],
            selected_label="Finance",
            selected_labels=["Finance"],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="Finance > Tax",
            selected_ids=["Finance > Tax"],
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
    assert result.final_node.label == "Tax"
    assert result.final_nodes is not None
    assert [node.label for node in result.final_nodes] == ["Tax"]
    assert [node.label for node in result.path_nodes] == ["Finance", "Tax"]
    assert result.stop_reason is ClassificationStopReason.STOP
    assert len(result.path) == 2


@pytest.mark.anyio
async def test_classifier_traverses_multiple_branches():
    """Classifier should recurse into multiple selected branches."""
    meat = TaxonomyNode(
        label="Meat",
        children=[TaxonomyNode(label="Beef")],
    )
    vegetables = TaxonomyNode(
        label="Vegetables",
        children=[TaxonomyNode(label="Carrot")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[meat, vegetables])

    steps = [
        ClassificationStep(
            selected_ids=["Meat", "Vegetables"],
            selected_labels=["Meat", "Vegetables"],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_ids=["Meat > Beef"],
            selected_labels=["Beef"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
        ClassificationStep(
            selected_ids=["Vegetables > Carrot"],
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
    assert [node.label for node in result.final_nodes] == ["Beef", "Carrot"]
    assert [node.label for node in result.path_nodes] == [
        "Meat",
        "Vegetables",
        "Beef",
        "Carrot",
    ]
    assert result.path[-1].selected_ids == ["Vegetables > Carrot"]


@pytest.mark.anyio
async def test_classifier_single_class_limits_branches():
    """Classifier should limit traversal to a single branch when enabled."""
    meat = TaxonomyNode(
        label="Meat",
        children=[TaxonomyNode(label="Beef")],
    )
    vegetables = TaxonomyNode(
        label="Vegetables",
        children=[TaxonomyNode(label="Carrot")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[meat, vegetables])

    steps = [
        ClassificationStep(
            selected_ids=["Meat", "Vegetables"],
            selected_labels=["Meat", "Vegetables"],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_ids=["Meat > Beef"],
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
    assert [node.label for node in result.final_nodes] == ["Beef"]
    assert [node.label for node in result.path_nodes] == ["Meat", "Beef"]


@pytest.mark.anyio
async def test_classifier_confidence_threshold_stops_branch():
    """Classifier should stop a branch when confidence is below the threshold."""
    root = TaxonomyNode(
        label="Root",
        children=[TaxonomyNode(label="Child")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    step = ClassificationStep(
        selected_ids=["Root"],
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

    root = TaxonomyNode(label="Root")
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    step = ClassificationStep(
        selected_id="Root",
        selected_ids=["Root"],
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
    assert result.final_node.label == "Root"


@pytest.mark.anyio
async def test_classifier_falls_back_when_selected_ids_empty():
    """Classifier should fall back to selected_id when selected_ids is empty."""

    root = TaxonomyNode(
        label="Finance",
        children=[TaxonomyNode(label="Tax")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    steps = [
        ClassificationStep(
            selected_id="Finance",
            selected_ids=[""],
            selected_label="Finance",
            selected_labels=None,
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="Finance > Tax",
            selected_ids=["Finance > Tax"],
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
async def test_classifier_falls_back_when_selected_labels_empty():
    """Classifier should fall back to selected_label when selected_labels is empty."""

    root = TaxonomyNode(
        label="Finance",
        children=[TaxonomyNode(label="Tax")],
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
            selected_id="Finance > Tax",
            selected_ids=["Finance > Tax"],
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


def test_classifier_path_map_disambiguates_duplicate_labels() -> None:
    """Classifier should disambiguate duplicate labels at the same level."""
    nodes = [
        TaxonomyNode(label="Duplicate"),
        TaxonomyNode(label="Duplicate"),
    ]

    node_paths = _build_node_path_map(nodes, [])

    assert list(node_paths.keys()) == ["Duplicate", "Duplicate (2)"]

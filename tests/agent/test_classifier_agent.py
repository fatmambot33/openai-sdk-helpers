from __future__ import annotations

from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent.classifier import (
    TaxonomyClassifierAgent,
    _build_node_path_map,
    _build_step_structure,
    _normalize_step_output,
)
from openai_sdk_helpers.structure import (
    ClassificationResult,
    ClassificationStopReason,
    ClassificationStep,
    StructureBase,
    Taxonomy,
    TaxonomyNode,
)


def test_classifier_default_prompt_template():
    """Classifier should use the bundled classifier prompt by default."""

    agent = TaxonomyClassifierAgent(
        model="gpt-4o-mini", taxonomy=TaxonomyNode(label="Root")
    )

    prompt = agent._build_prompt_from_jinja()
    assert "taxonomy classification assistant" in prompt


def _enum_member(enum_cls: type[Enum], value: str) -> Enum:
    """Return enum member matching the provided value."""
    return enum_cls._value2member_map_[value]


def _build_step(values: list[str]) -> tuple[type[ClassificationStep], type[Enum]]:
    """Build a step structure and its enum class for provided values."""
    step_structure = _build_step_structure(values)
    enum_cls = step_structure._extract_enum_class(
        step_structure.model_fields["selected_nodes"].annotation
    )
    assert enum_cls is not None
    return step_structure, enum_cls


@pytest.mark.anyio
async def test_classifier_traverses_taxonomy_levels():
    """Classifier should walk the taxonomy until a terminal step."""
    root = TaxonomyNode(
        label="Finance",
        children=[TaxonomyNode(label="Tax")],
    )
    alternate = TaxonomyNode(label="Health")
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root, alternate])

    root_step, root_enum = _build_step(["Finance", "Health"])
    tax_step, tax_enum = _build_step(["Finance > Tax"])
    steps = [
        root_step(
            selected_nodes=[_enum_member(root_enum, "Finance")],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        tax_step(
            selected_nodes=[_enum_member(tax_enum, "Finance > Tax")],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "_run_step_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_async("Tax update")

    assert isinstance(result, ClassificationResult)
    assert result.final_node is not None
    assert result.final_node.label == "Tax"
    assert result.final_nodes is not None
    assert [node.label for node in result.final_nodes] == ["Tax"]
    assert result.stop_reason is ClassificationStopReason.STOP
    assert len(result.steps) == 2


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

    root_step, root_enum = _build_step(["Meat", "Vegetables"])
    meat_step, meat_enum = _build_step(["Meat > Beef"])
    veg_step, veg_enum = _build_step(["Vegetables > Carrot"])
    steps = [
        root_step(
            selected_nodes=[
                _enum_member(root_enum, "Meat"),
                _enum_member(root_enum, "Vegetables"),
            ],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        meat_step(
            selected_nodes=[_enum_member(meat_enum, "Meat > Beef")],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
        veg_step(
            selected_nodes=[_enum_member(veg_enum, "Vegetables > Carrot")],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "_run_step_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_async("Culinary update")

    assert result.final_nodes is not None
    assert [node.label for node in result.final_nodes] == ["Beef", "Carrot"]
    assert result.steps[-1].selected_nodes == [
        _enum_member(veg_enum, "Vegetables > Carrot")
    ]


@pytest.mark.anyio
async def test_classifier_avoids_duplicate_leaf_nodes() -> None:
    """Classifier should avoid duplicating leaf nodes when merging branches."""
    leaf = TaxonomyNode(label="Leaf")
    branch = TaxonomyNode(
        label="Branch",
        children=[TaxonomyNode(label="Child")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[leaf, branch])

    root_step, root_enum = _build_step(["Leaf", "Branch"])
    child_step, child_enum = _build_step(["Branch > Child"])
    steps = [
        root_step(
            selected_nodes=[
                _enum_member(root_enum, "Leaf"),
                _enum_member(root_enum, "Branch"),
            ],
            confidence=0.7,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        child_step(
            selected_nodes=[_enum_member(child_enum, "Branch > Child")],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "_run_step_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.side_effect = steps
        result = await agent.run_async("Mixed taxonomy")

    assert result.final_nodes is not None
    assert [node.label for node in result.final_nodes] == ["Leaf", "Child"]


@pytest.mark.anyio
async def test_classifier_confidence_threshold_stops_branch():
    """Classifier should stop a branch when confidence is below the threshold."""
    root = TaxonomyNode(
        label="Root",
        children=[TaxonomyNode(label="Child")],
    )
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    root_step, root_enum = _build_step(["Root"])
    step = root_step(
        selected_nodes=[_enum_member(root_enum, "Root")],
        confidence=0.2,
        stop_reason=ClassificationStopReason.CONTINUE,
    )

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "_run_step_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = step
        result = await agent.run_async("Low confidence", confidence_threshold=0.5)

    assert result.stop_reason is ClassificationStopReason.NO_MATCH
    assert result.final_nodes is None


@pytest.mark.anyio
async def test_classifier_stops_when_no_children():
    """Classifier should stop when a selected node has no children."""

    root = TaxonomyNode(label="Root")
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[root])

    root_step, root_enum = _build_step(["Root"])
    step = root_step(
        selected_nodes=[_enum_member(root_enum, "Root")],
        confidence=0.6,
        stop_reason=ClassificationStopReason.CONTINUE,
    )

    with (
        patch.object(agent, "get_agent", return_value=MagicMock()),
        patch.object(agent, "_run_step_async", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = step
        result = await agent.run_async("Root only")

    assert result.stop_reason is ClassificationStopReason.NO_CHILDREN
    assert result.final_node is not None
    assert result.final_node.label == "Root"


@pytest.mark.anyio
async def test_classifier_requires_taxonomy_nodes():
    """Classifier should reject empty taxonomy definitions."""

    with pytest.raises(ValueError, match="taxonomy must include at least one node"):
        TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[])


def test_classifier_uses_taxonomy_children_as_roots() -> None:
    """Classifier should treat Taxonomy children as root nodes."""
    taxonomy = Taxonomy(label="Root", children=[TaxonomyNode(label="Leaf")])
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=taxonomy)

    assert [node.label for node in agent.root_nodes] == ["Leaf"]


def test_classifier_path_map_disambiguates_duplicate_labels() -> None:
    """Classifier should disambiguate duplicate labels at the same level."""
    nodes = [
        TaxonomyNode(label="Duplicate"),
        TaxonomyNode(label="Duplicate"),
    ]

    node_paths = _build_node_path_map(nodes, [])

    assert list(node_paths.keys()) == ["Duplicate", "Duplicate (2)"]


@pytest.mark.anyio
async def test_classifier_builds_sub_agents() -> None:
    """Classifier should construct sub-agents for child taxonomy nodes."""
    taxonomy = [
        TaxonomyNode(
            label="Root",
            children=[TaxonomyNode(label="Child")],
        )
    ]
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=taxonomy)
    root_step, root_enum = _build_step(["Root"])
    child_step, child_enum = _build_step(["Root > Child"])
    steps = [
        root_step(
            selected_nodes=[_enum_member(root_enum, "Root")],
            confidence=0.8,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        child_step(
            selected_nodes=[_enum_member(child_enum, "Root > Child")],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(
            TaxonomyClassifierAgent,
            "_run_step_async",
            new_callable=AsyncMock,
        ) as mock_run,
        patch.object(
            agent, "_build_sub_agent", wraps=agent._build_sub_agent
        ) as mock_build,
    ):
        mock_run.side_effect = steps
        result = await agent.run_async("delegate this")

    assert mock_build.call_count == 1
    assert mock_build.call_args.args[0] == taxonomy[0].children
    assert mock_run.call_count == 2
    assert result.final_nodes is not None
    assert [node.label for node in result.final_nodes] == ["Child"]


def test_classifier_maps_enum_selections_to_identifiers() -> None:
    """Classifier should normalize enum selections into identifiers."""
    nodes = [TaxonomyNode(label="Billing")]
    node_paths = _build_node_path_map(nodes, [])
    step_structure = _build_step_structure(list(node_paths.keys()))
    enum_cls = step_structure._extract_enum_class(
        step_structure.model_fields["selected_nodes"].annotation
    )
    assert enum_cls is not None
    enum_member = list(enum_cls)[0]
    assert enum_member.value == "Billing"
    raw_step = step_structure(
        selected_nodes=[enum_member],
        confidence=0.7,
        stop_reason=ClassificationStopReason.STOP,
    )

    normalized = _normalize_step_output(raw_step, step_structure)

    assert normalized.selected_nodes == [enum_member]


@pytest.mark.anyio
async def test_classifier_run_async_delegates_to_run_agent() -> None:
    """Classifier run_async should delegate to _run_agent."""
    agent = TaxonomyClassifierAgent(
        model="gpt-4o-mini", taxonomy=TaxonomyNode(label="Root")
    )
    expected = ClassificationResult(stop_reason=ClassificationStopReason.STOP)

    with patch.object(agent, "_run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = expected
        result = await agent.run_async(
            "Tax update",
            context={"source": "unit-test"},
            max_depth=1,
            confidence_threshold=0.4,
        )

    assert result is expected
    mock_run.assert_awaited_once_with(
        "Tax update",
        context={"source": "unit-test"},
        file_ids=None,
        max_depth=1,
        confidence_threshold=0.4,
    )


@pytest.mark.anyio
async def test_classifier_attaches_file_ids_to_steps() -> None:
    """Classifier should reuse file IDs across recursive steps."""
    taxonomy = [
        TaxonomyNode(
            label="Root",
            children=[TaxonomyNode(label="Child")],
        )
    ]
    agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=taxonomy)
    root_step, root_enum = _build_step(["Root"])
    child_step, child_enum = _build_step(["Root > Child"])
    steps = [
        root_step(
            selected_nodes=[_enum_member(root_enum, "Root")],
            confidence=0.8,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        child_step(
            selected_nodes=[_enum_member(child_enum, "Root > Child")],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]
    inputs: list[object] = []

    async def fake_run_step(*, input, **kwargs) -> StructureBase:
        inputs.append(input)
        return steps[len(inputs) - 1]

    with patch.object(agent, "_run_step_async", new=fake_run_step):
        result = await agent.run_async("Attach file", file_ids="file_123")

    assert result.final_node is not None
    assert [node.label for node in result.final_nodes or []] == ["Child"]
    assert len(inputs) == 2
    for payload in inputs:
        assert isinstance(payload, list)
        assert payload[0]["content"][1]["file_id"] == "file_123"


def test_classifier_run_sync_delegates_to_run_agent() -> None:
    """Classifier run_sync should delegate to _run_agent."""
    agent = TaxonomyClassifierAgent(
        model="gpt-4o-mini", taxonomy=TaxonomyNode(label="Root")
    )
    expected = ClassificationResult(stop_reason=ClassificationStopReason.STOP)

    with patch.object(agent, "_run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = expected
        result = agent.run_sync(
            "Tax update",
            context={"source": "unit-test"},
            max_depth=2,
            confidence_threshold=0.5,
        )

    assert result is expected
    mock_run.assert_awaited_once_with(
        "Tax update",
        context={"source": "unit-test"},
        file_ids=None,
        max_depth=2,
        confidence_threshold=0.5,
    )


@pytest.mark.anyio
async def test_classifier_run_sync_raises_thread_errors() -> None:
    """Classifier run_sync should re-raise thread errors."""
    agent = TaxonomyClassifierAgent(
        model="gpt-4o-mini", taxonomy=TaxonomyNode(label="Root")
    )

    with patch.object(agent, "_run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = ValueError("Boom")
        with pytest.raises(ValueError, match="Boom"):
            agent.run_sync("Tax update")

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from openai_sdk_helpers.agent.classifier_v2 import (
    TaxonomyClassifierAgentV2,
    _build_node_path_map,
    _build_step_structure_v2,
    _normalize_step_output,
)
from openai_sdk_helpers.structure import (
    ClassificationStep,
    ClassificationStopReason,
    TaxonomyNode,
)


@pytest.mark.anyio
async def test_classifier_v2_builds_sub_agents() -> None:
    """Classifier V2 should construct sub-agents for child taxonomy nodes."""
    taxonomy = [
        TaxonomyNode(
            label="Root",
            children=[TaxonomyNode(label="Child")],
        )
    ]
    agent = TaxonomyClassifierAgentV2(model="gpt-4o-mini", taxonomy=taxonomy)
    steps = [
        ClassificationStep(
            selected_id="Root",
            selected_ids=["Root"],
            selected_label="Root",
            selected_labels=["Root"],
            confidence=0.8,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="Root > Child",
            selected_ids=["Root > Child"],
            selected_label="Child",
            selected_labels=["Child"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    with (
        patch.object(
            TaxonomyClassifierAgentV2,
            "run_async",
            new_callable=AsyncMock,
        ) as mock_run,
        patch.object(
            agent, "_build_sub_agent", wraps=agent._build_sub_agent
        ) as mock_build,
    ):
        mock_run.side_effect = steps
        result = await agent.run_agent("delegate this")

    assert mock_build.call_count == 1
    assert mock_build.call_args.args[0] == taxonomy[0].children
    assert mock_run.call_count == 2
    assert result.final_nodes is not None
    assert [node.label for node in result.final_nodes] == ["Child"]


def test_classifier_v2_maps_step_v2_fields_to_selected_ids() -> None:
    """Classifier V2 should map enum selections into identifier fields."""
    nodes = [TaxonomyNode(label="Billing")]
    node_paths = _build_node_path_map(nodes, [])
    step_structure = _build_step_structure_v2(list(node_paths.keys()))
    enum_cls = step_structure._extract_enum_class(
        step_structure.model_fields["selected_node"].annotation
    )
    assert enum_cls is not None
    enum_member = list(enum_cls)[0]
    assert enum_member.value == "Billing"
    raw_step = step_structure(
        selected_node=enum_member,
        selected_nodes=[enum_member],
        confidence=0.7,
        stop_reason=ClassificationStopReason.STOP,
    )

    normalized = _normalize_step_output(raw_step, step_structure)

    assert normalized.selected_id == "Billing"
    assert normalized.selected_ids == ["Billing"]

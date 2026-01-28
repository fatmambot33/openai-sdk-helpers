from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from openai_sdk_helpers.agent.classifier_v2 import TaxonomyClassifierAgentV2
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
            id="root",
            label="Root",
            children=[TaxonomyNode(id="child", label="Child")],
        )
    ]
    agent = TaxonomyClassifierAgentV2(model="gpt-4o-mini", taxonomy=taxonomy)
    steps = [
        ClassificationStep(
            selected_id="root",
            selected_ids=["root"],
            selected_label="Root",
            selected_labels=["Root"],
            confidence=0.8,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="child",
            selected_ids=["child"],
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
    assert [node.id for node in result.final_nodes] == ["child"]

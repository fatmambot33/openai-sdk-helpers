from __future__ import annotations

from openai_sdk_helpers.structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    TaxonomyNode,
)


def test_taxonomy_node_build_path():
    """TaxonomyNode should build a computed path from parents."""

    node = TaxonomyNode(label="Leaf")

    assert node.computed_path == ["Leaf"]
    assert node.is_leaf is True
    assert node.build_path(["Root", "Branch"]) == ["Root", "Branch", "Leaf"]


def test_classification_result_properties():
    """ClassificationResult should expose computed properties."""

    root_node = TaxonomyNode(label="Root")
    leaf_node = TaxonomyNode(label="Leaf")
    branch_node = TaxonomyNode(label="Branch")

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
            selected_id="Root > Leaf",
            selected_ids=["Root > Leaf", "Root > Branch"],
            selected_label="Leaf",
            selected_labels=["Leaf", "Branch"],
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    result = ClassificationResult(
        final_node=leaf_node,
        final_nodes=[leaf_node, branch_node],
        confidence=0.9,
        stop_reason=ClassificationStopReason.STOP,
        path=steps,
        path_nodes=[root_node, leaf_node, branch_node],
    )

    assert result.depth == 2
    assert result.path_labels == ["Root", "Leaf", "Branch"]
    assert result.final_node == leaf_node
    assert result.final_nodes == [leaf_node, branch_node]
    assert [node.label for node in result.path_nodes] == [
        "Root",
        "Leaf",
        "Branch",
    ]


def test_stop_reason_is_terminal_property():
    """ClassificationStopReason should expose terminal state as a property."""

    assert ClassificationStopReason.NO_MATCH.is_terminal is True
    assert ClassificationStopReason.CONTINUE.is_terminal is False

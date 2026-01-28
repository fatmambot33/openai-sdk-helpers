from __future__ import annotations

from openai_sdk_helpers.structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    TaxonomyNode,
)


def test_taxonomy_node_build_path():
    """TaxonomyNode should build a computed path from parents."""

    node = TaxonomyNode(id="leaf", label="Leaf")

    assert node.computed_path == ["Leaf"]
    assert node.is_leaf is True
    assert node.build_path(["Root", "Branch"]) == ["Root", "Branch", "Leaf"]


def test_classification_result_properties():
    """ClassificationResult should expose computed properties."""

    steps = [
        ClassificationStep(
            selected_id="root",
            selected_label="Root",
            confidence=0.8,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        ClassificationStep(
            selected_id="leaf",
            selected_label="Leaf",
            confidence=0.9,
            stop_reason=ClassificationStopReason.STOP,
        ),
    ]

    result = ClassificationResult(
        final_id="leaf",
        final_label="Leaf",
        confidence=0.9,
        stop_reason=ClassificationStopReason.STOP,
        path=steps,
    )

    assert result.depth == 2
    assert result.path_labels == ["Root", "Leaf"]


def test_stop_reason_is_terminal_property():
    """ClassificationStopReason should expose terminal state as a property."""

    assert ClassificationStopReason.NO_MATCH.is_terminal is True
    assert ClassificationStopReason.CONTINUE.is_terminal is False

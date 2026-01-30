from __future__ import annotations

from enum import Enum

from openai_sdk_helpers.structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    Taxonomy,
    TaxonomyNode,
    taxonomy_enum_path,
)


def test_taxonomy_node_build_path():
    """TaxonomyNode should build a computed path from parents."""

    node = TaxonomyNode(label="Leaf")

    assert node.computed_path == ["Leaf"]
    assert node.is_leaf is True
    assert node.build_path(["Root", "Branch"]) == ["Root", "Branch", "Leaf"]


def test_taxonomy_flattened_nodes():
    """Taxonomy should return flattened taxonomy nodes."""

    leaf = TaxonomyNode(label="Leaf")
    branch = TaxonomyNode(label="Branch", children=[leaf])
    taxonomy = Taxonomy(
        name="Support",
        description="Customer support taxonomy.",
        nodes=[branch],
    )

    assert taxonomy.name == "Support"
    assert taxonomy.description == "Customer support taxonomy."
    assert [node.label for node in taxonomy.flattened_nodes] == ["Branch", "Leaf"]


def test_classification_result_properties():
    """ClassificationResult should expose computed properties."""

    root_node = TaxonomyNode(label="Root")
    leaf_node = TaxonomyNode(label="Leaf")
    branch_node = TaxonomyNode(label="Branch")

    step_enum = Enum(
        "StepEnum",
        {
            "ROOT": "Root",
            "ROOT_LEAF": "Root > Leaf",
            "ROOT_BRANCH": "Root > Branch",
        },
    )
    Step = ClassificationStep.build_for_enum(step_enum)
    steps = [
        Step(
            selected_node=step_enum.ROOT,
            selected_nodes=[step_enum.ROOT],
            confidence=0.8,
            stop_reason=ClassificationStopReason.CONTINUE,
        ),
        Step(
            selected_node=step_enum.ROOT_LEAF,
            selected_nodes=[step_enum.ROOT_LEAF, step_enum.ROOT_BRANCH],
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
    assert result.path_identifiers == ["Root", "Root > Leaf", "Root > Branch"]
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


def test_taxonomy_enum_path():
    """taxonomy_enum_path should return segments from enum values."""

    step_enum = Enum(
        "StepEnum",
        {
            "ROOT": "Root",
            "ROOT_LEAF": "Root > Leaf",
            "ROOT_ESCAPED": "Root > Leaf\\>Branch",
        },
    )

    assert taxonomy_enum_path(step_enum.ROOT) == ["Root"]
    assert taxonomy_enum_path(step_enum.ROOT_LEAF) == ["Root", "Leaf"]
    assert taxonomy_enum_path(step_enum.ROOT_ESCAPED) == ["Root", "Leaf > Branch"]
    assert taxonomy_enum_path("Root > Branch") == ["Root", "Branch"]
    assert taxonomy_enum_path(None) == []

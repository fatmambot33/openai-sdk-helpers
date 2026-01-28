from __future__ import annotations

from openai_sdk_helpers.structure import TaxonomyNode


def test_taxonomy_node_build_path():
    """TaxonomyNode should build a computed path from parents."""

    node = TaxonomyNode(id="leaf", label="Leaf")

    assert node.build_path() == ["Leaf"]
    assert node.build_path(["Root", "Branch"]) == ["Root", "Branch", "Leaf"]

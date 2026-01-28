"""Structured taxonomy and classification result models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Optional

from .base import StructureBase, spec_field


class TaxonomyNode(StructureBase):
    """Represent a taxonomy node with optional child categories.

    Attributes
    ----------
    id : str
        Unique identifier for the taxonomy node.
    label : str
        Human-readable label for the taxonomy node.
    description : str or None
        Optional description of the node.
    children : list[TaxonomyNode]
        Child nodes in the taxonomy.

    Methods
    -------
    build_path(parent_path)
        Build a computed path using the provided parent path segments.
    computed_path
        Return the computed path for the node.
    is_leaf
        Return True when the taxonomy node has no children.
    child_by_id(node_id)
        Return the child node matching the provided identifier.
    """

    id: str = spec_field("id", description="Unique identifier for the taxonomy.")
    label: str = spec_field(
        "label", description="Human-readable label for the taxonomy node."
    )
    description: Optional[str] = spec_field(
        "description",
        description="Optional description of the taxonomy node.",
        default=None,
    )
    children: list["TaxonomyNode"] = spec_field(
        "children",
        description="Child nodes in the taxonomy.",
        default_factory=list,
    )

    @property
    def is_leaf(self) -> bool:
        """Return True when the taxonomy node has no children.

        Returns
        -------
        bool
            True if the node has no children.
        """
        return not self.children

    def build_path(self, parent_path: Iterable[str] | None = None) -> list[str]:
        """Build a computed path using the provided parent path segments.

        Parameters
        ----------
        parent_path : Iterable[str] or None, default=None
            Parent path segments to prepend to the node label.

        Returns
        -------
        list[str]
            Computed path segments for this node.
        """
        if parent_path is None:
            return [self.label]
        return [*parent_path, self.label]

    @property
    def computed_path(self) -> list[str]:
        """Return the computed path for the node.

        Returns
        -------
        list[str]
            Computed path segments for this node.
        """
        return self.build_path()

    def child_by_id(self, node_id: str | None) -> Optional["TaxonomyNode"]:
        """Return the child node matching the provided identifier.

        Parameters
        ----------
        node_id : str or None
            Identifier of the child node to locate.

        Returns
        -------
        TaxonomyNode or None
            Matching child node, if found.
        """
        if node_id is None:
            return None
        return next((child for child in self.children if child.id == node_id), None)


class ClassificationStopReason(str, Enum):
    """Enumerate stop reasons for taxonomy classification.

    Methods
    -------
    is_terminal
        Return True if the stop reason should halt traversal.
    """

    CONTINUE = "continue"
    STOP = "stop"
    NO_MATCH = "no_match"
    MAX_DEPTH = "max_depth"
    NO_CHILDREN = "no_children"

    @property
    def is_terminal(self) -> bool:
        """Return True if the stop reason should halt traversal.

        Returns
        -------
        bool
            True when traversal should stop.
        """
        return self in {
            ClassificationStopReason.STOP,
            ClassificationStopReason.NO_MATCH,
            ClassificationStopReason.MAX_DEPTH,
            ClassificationStopReason.NO_CHILDREN,
        }


class ClassificationStep(StructureBase):
    """Represent a single classification step within a taxonomy level.

    Attributes
    ----------
    selected_id : str or None
        Identifier of the selected taxonomy node.
    selected_label : str or None
        Label of the selected taxonomy node.
    confidence : float or None
        Confidence score between 0 and 1.
    stop_reason : ClassificationStopReason
        Reason for stopping or continuing traversal.
    rationale : str or None
        Optional rationale for the classification decision.

    Methods
    -------
    as_summary()
        Return a dictionary summary of the classification step.
    """

    selected_id: Optional[str] = spec_field(
        "selected_id",
        description="Identifier of the selected taxonomy node.",
        default=None,
    )
    selected_label: Optional[str] = spec_field(
        "selected_label",
        description="Label of the selected taxonomy node.",
        default=None,
    )
    confidence: Optional[float] = spec_field(
        "confidence",
        description="Confidence score between 0 and 1.",
        default=None,
    )
    stop_reason: ClassificationStopReason = spec_field(
        "stop_reason",
        description="Reason for stopping or continuing traversal.",
        default=ClassificationStopReason.STOP,
    )
    rationale: Optional[str] = spec_field(
        "rationale",
        description="Optional rationale for the classification decision.",
        default=None,
    )

    def as_summary(self) -> dict[str, Any]:
        """Return a dictionary summary of the classification step.

        Returns
        -------
        dict[str, Any]
            Summary data for logging or inspection.
        """
        return {
            "selected_id": self.selected_id,
            "selected_label": self.selected_label,
            "confidence": self.confidence,
            "stop_reason": self.stop_reason.value,
        }


class ClassificationResult(StructureBase):
    """Represent the final result of taxonomy traversal.

    Attributes
    ----------
    final_id : str or None
        Identifier of the final taxonomy node selection.
    final_label : str or None
        Label of the final taxonomy node selection.
    confidence : float or None
        Confidence score for the final selection.
    stop_reason : ClassificationStopReason
        Reason the traversal ended.
    path : list[ClassificationStep]
        Ordered list of classification steps.

    Methods
    -------
    depth
        Return the number of classification steps recorded.
    path_labels
        Return the labels selected at each step.
    """

    final_id: Optional[str] = spec_field(
        "final_id",
        description="Identifier of the final taxonomy node selection.",
        default=None,
    )
    final_label: Optional[str] = spec_field(
        "final_label",
        description="Label of the final taxonomy node selection.",
        default=None,
    )
    confidence: Optional[float] = spec_field(
        "confidence",
        description="Confidence score for the final selection.",
        default=None,
    )
    stop_reason: ClassificationStopReason = spec_field(
        "stop_reason",
        description="Reason the traversal ended.",
        default=ClassificationStopReason.STOP,
    )
    path: list[ClassificationStep] = spec_field(
        "path",
        description="Ordered list of classification steps.",
        default_factory=list,
    )

    @property
    def depth(self) -> int:
        """Return the number of classification steps recorded.

        Returns
        -------
        int
            Count of classification steps.
        """
        return len(self.path)

    @property
    def path_labels(self) -> list[str]:
        """Return the labels selected at each step.

        Returns
        -------
        list[str]
            Labels selected at each classification step.
        """
        return [step.selected_label for step in self.path if step.selected_label]


def flatten_taxonomy(nodes: Iterable[TaxonomyNode]) -> list[TaxonomyNode]:
    """Return a flattened list of taxonomy nodes.

    Parameters
    ----------
    nodes : Iterable[TaxonomyNode]
        Root nodes to traverse.

    Returns
    -------
    list[TaxonomyNode]
        Depth-first ordered list of nodes.
    """
    flattened: list[TaxonomyNode] = []
    for node in nodes:
        flattened.append(node)
        if node.children:
            flattened.extend(flatten_taxonomy(node.children))
    return flattened


__all__ = [
    "ClassificationResult",
    "ClassificationStep",
    "ClassificationStopReason",
    "TaxonomyNode",
    "flatten_taxonomy",
]

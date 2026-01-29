"""Structured taxonomy and classification result models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Optional, cast

from .base import StructureBase, spec_field


class TaxonomyNode(StructureBase):
    """Represent a taxonomy node with optional child categories.

    Attributes
    ----------
    label : str
        Human-readable label for the taxonomy node.
    description : str | None
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
    child_by_path(path)
        Return the child node matching the provided path.
    """

    label: str = spec_field(
        "label", description="Human-readable label for the taxonomy node."
    )
    description: str | None = spec_field(
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

    def child_by_path(
        self, path: Iterable[str] | str | None
    ) -> Optional["TaxonomyNode"]:
        """Return the child node matching the provided path.

        Parameters
        ----------
        path : Iterable[str] or str or None
            Path segments or a delimited path string to locate.

        Returns
        -------
        TaxonomyNode or None
            Matching child node, if found.
        """
        if path is None:
            return None
        if isinstance(path, str):
            path_value = path
        else:
            path_value = " > ".join(path)
        last_segment = path_value.split(" > ")[-1] if path_value else None
        if not last_segment:
            return None
        return next(
            (child for child in self.children if child.label == last_segment),
            None,
        )


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
        Path identifier of the selected taxonomy node.
    selected_ids : list[str] or None
        Path identifiers of selected taxonomy nodes for multi-class classification.
    selected_label : str or None
        Label of the selected taxonomy node.
    selected_labels : list[str] or None
        Labels of selected taxonomy nodes for multi-class classification.
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

    Examples
    --------
    Create a multi-class step and summarize the selections:

    >>> step = ClassificationStep(
    ...     selected_ids=["billing", "invoicing"],
    ...     selected_labels=["Billing", "Invoicing"],
    ...     confidence=0.82,
    ...     stop_reason=ClassificationStopReason.STOP,
    ... )
    >>> step.as_summary()["selected_ids"]
    ['billing', 'invoicing']
    """

    selected_id: Optional[str] = spec_field(
        "selected_id",
        description="Path identifier of the selected taxonomy node.",
        default=None,
    )
    selected_ids: list[str] | None = spec_field(
        "selected_ids",
        description="Path identifiers of selected taxonomy nodes.",
        default=None,
    )
    selected_label: Optional[str] = spec_field(
        "selected_label",
        description="Label of the selected taxonomy node.",
        default=None,
    )
    selected_labels: list[str] | None = spec_field(
        "selected_labels",
        description="Labels of selected taxonomy nodes.",
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
        allow_null=False,
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

        Examples
        --------
        >>> step = ClassificationStep(selected_id="root", selected_label="Root")
        >>> step.as_summary()["selected_id"]
        'root'
        """
        return {
            "selected_id": self.selected_id,
            "selected_ids": self.selected_ids,
            "selected_label": self.selected_label,
            "selected_labels": self.selected_labels,
            "confidence": self.confidence,
            "stop_reason": self.stop_reason.value,
        }


class ClassificationStepV2(StructureBase):
    """Represent a classification step constrained to taxonomy node enums.

    Attributes
    ----------
    selected_node : Enum or None
        Enum value of the selected taxonomy node.
    selected_nodes : list[Enum] or None
        Enum values of selected taxonomy nodes for multi-class classification.
    confidence : float or None
        Confidence score between 0 and 1.
    stop_reason : ClassificationStopReason
        Reason for stopping or continuing traversal.
    rationale : str or None
        Optional rationale for the classification decision.

    Methods
    -------
    build_for_enum(enum_cls)
        Build a ClassificationStepV2 subclass with enum-constrained selections.
    as_summary()
        Return a dictionary summary of the classification step.

    Examples
    --------
    Create a multi-class V2 step and summarize the selections:

    >>> NodeEnum = Enum("NodeEnum", {"BILLING": "billing"})
    >>> StepEnum = ClassificationStepV2.build_for_enum(NodeEnum)
    >>> step = StepEnum(
    ...     selected_nodes=[NodeEnum.BILLING],
    ...     confidence=0.82,
    ...     stop_reason=ClassificationStopReason.STOP,
    ... )
    >>> step.as_summary()["selected_nodes"]
    [<NodeEnum.BILLING: 'billing'>]
    """

    selected_node: Enum | None = spec_field(
        "selected_node",
        description="Path identifier of the selected taxonomy node.",
        default=None,
    )
    selected_nodes: list[Enum] | None = spec_field(
        "selected_nodes",
        description="Path identifiers of selected taxonomy nodes.",
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
        allow_null=False,
    )
    rationale: Optional[str] = spec_field(
        "rationale",
        description="Optional rationale for the classification decision.",
        default=None,
    )

    @classmethod
    def build_for_enum(cls, enum_cls: type[Enum]) -> type["ClassificationStepV2"]:
        """Build a ClassificationStepV2 subclass with enum-constrained fields.

        Parameters
        ----------
        enum_cls : type[Enum]
            Enum type to use for node selections.

        Returns
        -------
        type[ClassificationStepV2]
            Specialized ClassificationStepV2 class bound to the enum.
        """
        namespace: dict[str, Any] = {
            "__annotations__": {
                "selected_node": enum_cls | None,
                "selected_nodes": list[enum_cls] | None,
            },
            "selected_node": spec_field(
                "selected_node",
                description="Path identifier of the selected taxonomy node.",
                default=None,
            ),
            "selected_nodes": spec_field(
                "selected_nodes",
                description="Path identifiers of selected taxonomy nodes.",
                default=None,
            ),
        }
        return cast(
            type["ClassificationStepV2"], type("BoundStepV2", (cls,), namespace)
        )

    def as_summary(self) -> dict[str, Any]:
        """Return a dictionary summary of the classification step.

        Returns
        -------
        dict[str, Any]
            Summary data for logging or inspection.

        Examples
        --------
        >>> NodeEnum = Enum("NodeEnum", {"ROOT": "root"})
        >>> StepEnum = ClassificationStepV2.build_for_enum(NodeEnum)
        >>> step = StepEnum(selected_node=NodeEnum.ROOT)
        >>> step.as_summary()["selected_node"]
        <NodeEnum.ROOT: 'root'>
        """
        return {
            "selected_node": self.selected_node,
            "selected_nodes": self.selected_nodes,
            "confidence": self.confidence,
            "stop_reason": self.stop_reason.value,
        }


class ClassificationResult(StructureBase):
    """Represent the final result of taxonomy traversal.

    Attributes
    ----------
    final_node : TaxonomyNode or None
        Resolved taxonomy node for the final selection.
    final_nodes : list[TaxonomyNode] or None
        Resolved taxonomy nodes for the final selections across branches.
    confidence : float or None
        Confidence score for the final selection.
    stop_reason : ClassificationStopReason
        Reason the traversal ended.
    path : list[ClassificationStep]
        Ordered list of classification steps.
    path_nodes : list[TaxonomyNode]
        Resolved taxonomy nodes selected across the path.

    Methods
    -------
    depth
        Return the number of classification steps recorded.
    path_labels
        Return the labels selected at each step.

    Examples
    --------
    Summarize single and multi-class output:

    >>> node = TaxonomyNode(label="Tax")
    >>> result = ClassificationResult(
    ...     final_node=node,
    ...     final_nodes=[node],
    ...     confidence=0.91,
    ...     stop_reason=ClassificationStopReason.STOP,
    ... )
    >>> result.final_nodes
    [TaxonomyNode(label='Tax', description=None, children=[])]
    """

    final_node: TaxonomyNode | None = spec_field(
        "final_node",
        description="Resolved taxonomy node for the final selection.",
        default=None,
    )
    final_nodes: list[TaxonomyNode] | None = spec_field(
        "final_nodes",
        description="Resolved taxonomy nodes for the final selections.",
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
    path_nodes: list[TaxonomyNode] = spec_field(
        "path_nodes",
        description="Resolved taxonomy nodes selected across the path.",
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

        Examples
        --------
        >>> steps = [
        ...     ClassificationStep(selected_label="Root"),
        ...     ClassificationStep(selected_labels=["Leaf", "Branch"]),
        ... ]
        >>> ClassificationResult(
        ...     stop_reason=ClassificationStopReason.STOP,
        ...     path=steps,
        ... ).path_labels
        ['Root', 'Leaf', 'Branch']
        """
        labels: list[str] = []
        for step in self.path:
            if step.selected_labels:
                labels.extend(step.selected_labels)
            elif step.selected_label:
                labels.append(step.selected_label)
        return labels


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
    "ClassificationStepV2",
    "ClassificationStopReason",
    "TaxonomyNode",
    "flatten_taxonomy",
]

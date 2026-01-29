"""Agent for taxonomy-driven text classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, cast

from ..structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    StructureBase,
    TaxonomyNode,
    spec_field,
)
from .base import AgentBase
from .configuration import AgentConfiguration


class TaxonomyClassifierAgent(AgentBase):
    """Classify text by traversing a taxonomy level by level.

    Parameters
    ----------
    template_path : Path | str | None, default=None
        Optional template file path for prompt rendering.
    model : str | None, default=None
        Model identifier to use for classification.

    Methods
    -------
    run_agent(text, taxonomy, context, max_depth)
        Classify text by walking the taxonomy tree.

    Examples
    --------
    Create a classifier with a flat taxonomy:

    >>> taxonomy = [
    ...     TaxonomyNode(label="Billing"),
    ...     TaxonomyNode(label="Support"),
    ... ]
    >>> agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=taxonomy)
    """

    def __init__(
        self,
        *,
        template_path: Path | str | None = None,
        model: str | None = None,
        taxonomy: TaxonomyNode | Sequence[TaxonomyNode],
    ) -> None:
        """Initialize the taxonomy classifier agent configuration.

        Parameters
        ----------
        template_path : Path | str | None, default=None
            Optional template file path for prompt rendering.
        model : str | None, default=None
            Model identifier to use for classification.

        Raises
        ------
        ValueError
            If the model is not provided.

        Examples
        --------
        >>> classifier = TaxonomyClassifierAgent(model="gpt-4o-mini")
        """
        self._taxonomy = taxonomy
        self._root_nodes = _normalize_roots(taxonomy)
        if not self._root_nodes:
            raise ValueError("taxonomy must include at least one node")
        resolved_template_path = template_path or _default_template_path()
        configuration = AgentConfiguration(
            name="taxonomy_classifier",
            instructions="Agent instructions",
            description="Classify text by traversing taxonomy levels.",
            template_path=resolved_template_path,
            output_structure=ClassificationStep,
            model=model,
        )
        super().__init__(configuration=configuration)

    async def run_agent(
        self,
        text: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        max_depth: Optional[int] = None,
        confidence_threshold: float | None = None,
        single_class: bool = False,
    ) -> ClassificationResult:
        """Classify ``text`` by iterating over taxonomy levels.

        Parameters
        ----------
        text : str
            Source text to classify.
        context : dict or None, default=None
            Additional context values to merge into the prompt.
        max_depth : int or None, default=None
            Maximum depth to traverse before stopping.
        confidence_threshold : float or None, default=None
            Minimum confidence required to accept a classification step.
        single_class : bool, default=False
            Whether to keep only the highest-priority selection per step.

        Returns
        -------
        ClassificationResult
            Structured classification result describing the traversal.

        Raises
        ------
        ValueError
            If the taxonomy is empty.

        Examples
        --------
        >>> taxonomy = TaxonomyNode(
        ...     label="Finance",
        ...     children=[TaxonomyNode(label="Tax")],
        ... )
        >>> agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=taxonomy)
        >>> isinstance(agent.root_nodes, list)
        True
        """
        path: list[ClassificationStep] = []
        path_nodes: list[TaxonomyNode] = []
        best_confidence: float | None = None
        stop_reason = ClassificationStopReason.NO_MATCH
        saw_max_depth = False
        saw_no_children = False
        saw_terminal_stop = False
        branch_queue = [
            _BranchState(nodes=list(self._root_nodes), depth=0, parent_path=[])
        ]
        final_nodes: list[TaxonomyNode] = []

        while branch_queue:
            branch = branch_queue.pop(0)
            current_nodes = branch.nodes
            depth = branch.depth
            if max_depth is not None and depth >= max_depth:
                saw_max_depth = True
                continue
            if not current_nodes:
                continue

            node_paths = _build_node_path_map(current_nodes, branch.parent_path)
            template_context = _build_context(
                node_descriptors=_build_node_descriptors(node_paths),
                path=path,
                depth=depth,
                context=context,
            )
            step_structure = _build_step_structure(
                list(node_paths.keys()), current_nodes
            )
            raw_step = await self.run_async(
                input=text,
                context=template_context,
                output_structure=step_structure,
            )
            step = _normalize_step_output(raw_step, step_structure)
            path.append(step)

            if (
                confidence_threshold is not None
                and step.confidence is not None
                and step.confidence < confidence_threshold
            ):
                continue

            resolved_nodes = _resolve_nodes(node_paths, step)
            if resolved_nodes:
                if single_class:
                    resolved_nodes = resolved_nodes[:1]
                path_nodes.extend(resolved_nodes)

            if step.stop_reason.is_terminal:
                if resolved_nodes:
                    final_nodes.extend(resolved_nodes)
                    best_confidence = _max_confidence(best_confidence, step.confidence)
                    saw_terminal_stop = True
                continue

            if not resolved_nodes:
                stop_reason = ClassificationStopReason.NO_MATCH
                continue

            for node in resolved_nodes:
                if node.children:
                    branch_queue.append(
                        _BranchState(
                            nodes=list(node.children),
                            depth=depth + 1,
                            parent_path=[*branch.parent_path, node.label],
                        )
                    )
                else:
                    saw_no_children = True
                    final_nodes.append(node)
                    best_confidence = _max_confidence(best_confidence, step.confidence)

        final_nodes_value = final_nodes or None
        final_node = final_nodes[0] if final_nodes else None
        if saw_terminal_stop:
            stop_reason = ClassificationStopReason.STOP
        elif final_nodes and saw_no_children:
            stop_reason = ClassificationStopReason.NO_CHILDREN
        elif final_nodes:
            stop_reason = ClassificationStopReason.STOP
        elif saw_max_depth:
            stop_reason = ClassificationStopReason.MAX_DEPTH
        elif saw_no_children:
            stop_reason = ClassificationStopReason.NO_CHILDREN
        return ClassificationResult(
            final_node=final_node,
            final_nodes=final_nodes_value,
            confidence=best_confidence,
            stop_reason=stop_reason,
            path=path,
            path_nodes=path_nodes,
        )

    @property
    def taxonomy(self) -> TaxonomyNode | Sequence[TaxonomyNode]:
        """Return the root taxonomy node(s).

        Returns
        -------
        TaxonomyNode or Sequence[TaxonomyNode]
            Root taxonomy node or list of root nodes.
        """
        return self._taxonomy

    @property
    def root_nodes(self) -> list[TaxonomyNode]:
        """Return the list of root taxonomy nodes.

        Returns
        -------
        list[TaxonomyNode]
            List of root taxonomy nodes.
        """
        return self._root_nodes


@dataclass(frozen=True)
class _BranchState:
    nodes: list[TaxonomyNode]
    depth: int
    parent_path: list[str]


def _normalize_roots(
    taxonomy: TaxonomyNode | Sequence[TaxonomyNode],
) -> list[TaxonomyNode]:
    """Normalize taxonomy input into a list of root nodes.

    Parameters
    ----------
    taxonomy : TaxonomyNode | Sequence[TaxonomyNode]
        Root taxonomy node or list of root nodes.

    Returns
    -------
    list[TaxonomyNode]
        Normalized list of root nodes.
    """
    if isinstance(taxonomy, TaxonomyNode):
        return [taxonomy]
    return [node for node in taxonomy if node is not None]


def _default_template_path() -> Path:
    """Return the built-in classifier prompt template path.

    Returns
    -------
    Path
        Path to the bundled classifier Jinja template.
    """
    return Path(__file__).resolve().parents[1] / "prompt" / "classifier.jinja"


def _build_context(
    *,
    node_descriptors: Iterable[dict[str, Any]],
    path: Sequence[ClassificationStep],
    depth: int,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the template context for a classification step.

    Parameters
    ----------
    node_descriptors : Iterable[dict[str, Any]]
        Node descriptors available at the current taxonomy level.
    path : Sequence[ClassificationStep]
        Steps recorded so far in the traversal.
    depth : int
        Current traversal depth.
    context : dict or None
        Optional additional context values.

    Returns
    -------
    dict[str, Any]
        Context dictionary for prompt rendering.
    """
    template_context: Dict[str, Any] = {
        "taxonomy_nodes": list(node_descriptors),
        "path": [step.as_summary() for step in path],
        "depth": depth,
    }
    if context:
        template_context.update(context)
    return template_context


def _build_step_structure(
    path_identifiers: Sequence[str],
    nodes: Sequence[TaxonomyNode],
) -> type[StructureBase]:
    """Build a step output structure constrained to taxonomy paths.

    Parameters
    ----------
    path_identifiers : Sequence[str]
        Path identifiers for nodes at the current classification step.
    nodes : Sequence[TaxonomyNode]
        Candidate taxonomy nodes for the current classification step.

    Returns
    -------
    type[StructureBase]
        Dynamic structure class for the classification step output.
    """
    id_enum = _build_taxonomy_enum("TaxonomyPath", path_identifiers)
    label_enum = _build_taxonomy_enum("TaxonomyLabel", [node.label for node in nodes])
    namespace: dict[str, Any] = {
        "__annotations__": {
            "selected_id": id_enum | None,
            "selected_ids": list[id_enum] | None,
            "selected_label": label_enum | None,
            "selected_labels": list[label_enum] | None,
            "confidence": float | None,
            "stop_reason": ClassificationStopReason,
            "rationale": str | None,
        },
        "selected_id": spec_field(
            "selected_id",
            description="Identifier of the selected taxonomy node.",
            default=None,
        ),
        "selected_ids": spec_field(
            "selected_ids",
            description="Identifiers of selected taxonomy nodes.",
            default=None,
        ),
        "selected_label": spec_field(
            "selected_label",
            description="Label of the selected taxonomy node.",
            default=None,
        ),
        "selected_labels": spec_field(
            "selected_labels",
            description="Labels of selected taxonomy nodes.",
            default=None,
        ),
        "confidence": spec_field(
            "confidence",
            description="Confidence score between 0 and 1.",
            default=None,
        ),
        "stop_reason": spec_field(
            "stop_reason",
            description="Reason for stopping or continuing traversal.",
            default=ClassificationStopReason.STOP,
            allow_null=False,
        ),
        "rationale": spec_field(
            "rationale",
            description="Optional rationale for the classification decision.",
            default=None,
        ),
    }
    step_structure = type(
        "TaxonomyStepStructure",
        (StructureBase,),
        namespace,
    )
    return cast(type[StructureBase], step_structure)


def _build_taxonomy_enum(name: str, values: Sequence[str]) -> type[Enum]:
    """Build a safe Enum from taxonomy node values.

    Parameters
    ----------
    name : str
        Name to use for the enum class.
    values : Sequence[str]
        Taxonomy node values to include as enum members.

    Returns
    -------
    type[Enum]
        Enum class with sanitized member names.
    """
    members: dict[str, str] = {}
    prefix = _sanitize_enum_prefix(name)
    for index, value in enumerate(values, start=1):
        member_name = _sanitize_enum_member(prefix, value, index, members)
        members[member_name] = value
    if not members:
        members["UNSPECIFIED"] = ""
    return cast(type[Enum], Enum(name, members))


def _build_node_path_map(
    nodes: Sequence[TaxonomyNode],
    parent_path: Sequence[str],
) -> dict[str, TaxonomyNode]:
    """Build a mapping of node path identifiers to taxonomy nodes.

    Parameters
    ----------
    nodes : Sequence[TaxonomyNode]
        Candidate nodes at the current taxonomy level.
    parent_path : Sequence[str]
        Path segments leading to the current taxonomy level.

    Returns
    -------
    dict[str, TaxonomyNode]
        Mapping of path identifiers to taxonomy nodes.
    """
    path_map: dict[str, TaxonomyNode] = {}
    for node in nodes:
        path = " > ".join([*parent_path, node.label])
        path_map[path] = node
    return path_map


def _build_node_descriptors(
    node_paths: dict[str, TaxonomyNode],
) -> list[dict[str, Any]]:
    """Build node descriptors for prompt rendering.

    Parameters
    ----------
    node_paths : dict[str, TaxonomyNode]
        Mapping of path identifiers to taxonomy nodes.

    Returns
    -------
    list[dict[str, Any]]
        Node descriptor dictionaries for prompt rendering.
    """
    descriptors: list[dict[str, Any]] = []
    for path_id, node in node_paths.items():
        descriptors.append(
            {
                "identifier": path_id,
                "label": node.label,
                "description": node.description,
            }
        )
    return descriptors


def _sanitize_enum_prefix(prefix: str) -> str:
    """Return a safe prefix for taxonomy enum member names.

    Parameters
    ----------
    prefix : str
        Prefix to normalize for enum member naming.

    Returns
    -------
    str
        Normalized prefix for enum members.
    """
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", prefix).strip("_").upper()
    return normalized or "VALUE"


def _sanitize_enum_member(
    prefix: str,
    value: str,
    index: int,
    existing: dict[str, str],
) -> str:
    """Return a valid enum member name for a taxonomy value.

    Parameters
    ----------
    prefix : str
        Enum member prefix to include in the name.
    value : str
        Raw taxonomy value to sanitize.
    index : int
        Index of the value in the source list.
    existing : dict[str, str]
        Existing enum members to avoid collisions.

    Returns
    -------
    str
        Sanitized enum member name.
    """
    normalized_value = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").upper()
    if not normalized_value:
        normalized_value = "VALUE"
    if normalized_value[0].isdigit():
        normalized_value = f"VALUE_{index}"
    normalized = f"{prefix}_{index}_{normalized_value}"
    candidate = normalized
    suffix = 1
    while candidate in existing:
        candidate = f"{normalized}_{suffix}"
        suffix += 1
    return candidate


def _normalize_step_output(
    step: StructureBase,
    step_structure: type[StructureBase],
) -> ClassificationStep:
    """Normalize dynamic step output into a ClassificationStep.

    Parameters
    ----------
    step : StructureBase
        Raw step output returned by the agent.
    step_structure : type[StructureBase]
        Structure definition used to parse the agent output.

    Returns
    -------
    ClassificationStep
        Normalized classification step instance.
    """
    payload = step.to_json()
    enum_fields = _extract_enum_fields(step_structure)
    normalized = {}
    for key, value in payload.items():
        enum_cls = enum_fields.get(key)
        if enum_cls is not None:
            normalized[key] = _normalize_enum_value(value, enum_cls)
        else:
            normalized[key] = value
    return ClassificationStep.from_json(normalized)


def _extract_enum_fields(
    step_structure: type[StructureBase],
) -> dict[str, type[Enum]]:
    """Return the enum field mapping for a step structure.

    Parameters
    ----------
    step_structure : type[StructureBase]
        Structure definition to inspect.

    Returns
    -------
    dict[str, type[Enum]]
        Mapping of field names to enum classes.
    """
    enum_fields: dict[str, type[Enum]] = {}
    for field_name, model_field in step_structure.model_fields.items():
        enum_cls = step_structure._extract_enum_class(model_field.annotation)
        if enum_cls is not None:
            enum_fields[field_name] = enum_cls
    return enum_fields


def _normalize_enum_value(value: Any, enum_cls: type[Enum]) -> Any:
    """Normalize enum values into raw primitives.

    Parameters
    ----------
    value : Any
        Value to normalize.
    enum_cls : type[Enum]
        Enum type used for normalization.

    Returns
    -------
    Any
        Primitive value suitable for ``ClassificationStep``.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_normalize_enum_value(item, enum_cls) for item in value]
    if isinstance(value, str):
        if value in enum_cls.__members__:
            return enum_cls.__members__[value].value
        if value in enum_cls._value2member_map_:
            return enum_cls(value).value
    return value


def _resolve_nodes(
    node_paths: dict[str, TaxonomyNode],
    step: ClassificationStep,
) -> list[TaxonomyNode]:
    """Resolve selected taxonomy nodes for a classification step.

    Parameters
    ----------
    node_paths : dict[str, TaxonomyNode]
        Mapping of path identifiers to nodes at the current level.
    step : ClassificationStep
        Classification step output to resolve.

    Returns
    -------
    list[TaxonomyNode]
        Matching taxonomy nodes in priority order.
    """
    resolved: list[TaxonomyNode] = []
    selected_ids = _selected_ids(step)
    if selected_ids:
        for selected_id in selected_ids:
            node = node_paths.get(selected_id)
            if node:
                resolved.append(node)
        if resolved:
            return resolved
    selected_labels = _selected_labels(step)
    if selected_labels:
        for selected_label in selected_labels:
            for node in node_paths.values():
                if node.label == selected_label:
                    resolved.append(node)
    return resolved


def _selected_ids(step: ClassificationStep) -> list[str]:
    """Return selected identifiers for a classification step.

    Parameters
    ----------
    step : ClassificationStep
        Classification output to normalize.

    Returns
    -------
    list[str]
        Selected identifiers in priority order.
    """
    if step.selected_ids is not None:
        selected_ids = [selected_id for selected_id in step.selected_ids if selected_id]
        if selected_ids:
            return selected_ids
    return [step.selected_id] if step.selected_id else []


def _selected_labels(step: ClassificationStep) -> list[str]:
    """Return selected labels for a classification step.

    Parameters
    ----------
    step : ClassificationStep
        Classification output to normalize.

    Returns
    -------
    list[str]
        Selected labels in priority order.
    """
    if step.selected_labels is not None:
        selected_labels = [
            selected_label for selected_label in step.selected_labels if selected_label
        ]
        if selected_labels:
            return selected_labels
    return [step.selected_label] if step.selected_label else []


def _max_confidence(
    current: float | None,
    candidate: float | None,
) -> float | None:
    """Return the higher confidence value.

    Parameters
    ----------
    current : float or None
        Current best confidence value.
    candidate : float or None
        Candidate confidence value to compare.

    Returns
    -------
    float or None
        Highest confidence value available.
    """
    if current is None:
        return candidate
    if candidate is None:
        return current
    return max(current, candidate)


__all__ = ["TaxonomyClassifierAgent"]

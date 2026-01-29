"""Recursive agent for taxonomy-driven text classification."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, cast

from ..structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    StructureBase,
    TaxonomyNode,
)
from .base import AgentBase
from .configuration import AgentConfiguration


class TaxonomyClassifierAgent(AgentBase):
    """Classify text by recursively traversing a taxonomy.

    Parameters
    ----------
    template_path : Path | str | None, default=None
        Optional template file path for prompt rendering.
    model : str | None, default=None
        Model identifier to use for classification.

    Methods
    -------
    run_agent(text, taxonomy, context, max_depth)
        Classify text by recursively walking the taxonomy tree.

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
        taxonomy : TaxonomyNode | Sequence[TaxonomyNode]
            Root taxonomy node or list of root nodes.

        Raises
        ------
        ValueError
            If the taxonomy is empty.

        Examples
        --------
        >>> classifier = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=[])
        """
        self._taxonomy = taxonomy
        self._root_nodes = _normalize_roots(taxonomy)
        if not self._root_nodes:
            raise ValueError("taxonomy must include at least one node")
        resolved_template_path = template_path or _default_template_path()
        configuration = AgentConfiguration(
            name="taxonomy_classifier",
            instructions="Agent instructions",
            description="Classify text by traversing taxonomy levels recursively.",
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
        """Classify ``text`` by recursively walking taxonomy levels.

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

        Examples
        --------
        >>> taxonomy = TaxonomyNode(label="Finance")
        >>> agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=taxonomy)
        >>> isinstance(agent.root_nodes, list)
        True
        """
        state = _TraversalState()
        await self._classify_nodes(
            text=text,
            nodes=list(self._root_nodes),
            depth=0,
            parent_path=[],
            context=context,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            single_class=single_class,
            state=state,
        )

        final_nodes_value = state.final_nodes or None
        final_node = state.final_nodes[0] if state.final_nodes else None
        stop_reason = _resolve_stop_reason(state)
        return ClassificationResult(
            final_node=final_node,
            final_nodes=final_nodes_value,
            confidence=state.best_confidence,
            stop_reason=stop_reason,
            path=state.path,
            path_nodes=state.path_nodes,
        )

    async def _classify_nodes(
        self,
        *,
        text: str,
        nodes: list[TaxonomyNode],
        depth: int,
        parent_path: list[str],
        context: Optional[Dict[str, Any]],
        max_depth: Optional[int],
        confidence_threshold: float | None,
        single_class: bool,
        state: "_TraversalState",
    ) -> None:
        """Classify a taxonomy level and recursively traverse children.

        Parameters
        ----------
        text : str
            Source text to classify.
        nodes : list[TaxonomyNode]
            Candidate taxonomy nodes for the current level.
        depth : int
            Current traversal depth.
        context : dict or None
            Additional context values to merge into the prompt.
        max_depth : int or None
            Maximum traversal depth before stopping.
        confidence_threshold : float or None
            Minimum confidence required to accept a classification step.
        single_class : bool
            Whether to keep only the highest-priority selection per step.
        state : _TraversalState
            Aggregated traversal state.
        """
        if max_depth is not None and depth >= max_depth:
            state.saw_max_depth = True
            return
        if not nodes:
            return

        node_paths = _build_node_path_map(nodes, parent_path)
        template_context = _build_context(
            node_descriptors=_build_node_descriptors(node_paths),
            path=state.path,
            depth=depth,
            context=context,
        )
        step_structure = _build_step_structure(list(node_paths.keys()))
        raw_step = await self.run_async(
            input=text,
            context=template_context,
            output_structure=step_structure,
        )
        step = _normalize_step_output(raw_step, step_structure)
        state.path.append(step)

        if (
            confidence_threshold is not None
            and step.confidence is not None
            and step.confidence < confidence_threshold
        ):
            return

        resolved_nodes = _resolve_nodes(node_paths, step)
        if resolved_nodes:
            if single_class:
                resolved_nodes = resolved_nodes[:1]
            state.path_nodes.extend(resolved_nodes)

        if step.stop_reason.is_terminal:
            if resolved_nodes:
                state.final_nodes.extend(resolved_nodes)
                state.best_confidence = _max_confidence(
                    state.best_confidence, step.confidence
                )
                state.saw_terminal_stop = True
            return

        if not resolved_nodes:
            return

        base_path_len = len(state.path)
        base_path_nodes_len = len(state.path_nodes)
        base_final_nodes_len = len(state.final_nodes)
        child_tasks = []
        for node in resolved_nodes:
            if node.children:
                sub_agent = self._build_sub_agent(list(node.children))
                sub_state = _copy_traversal_state(state)
                child_tasks.append(
                    self._classify_subtree(
                        sub_agent=sub_agent,
                        text=text,
                        nodes=list(node.children),
                        depth=depth + 1,
                        parent_path=[*parent_path, node.label],
                        context=context,
                        max_depth=max_depth,
                        confidence_threshold=confidence_threshold,
                        single_class=single_class,
                        state=sub_state,
                    )
                )
            else:
                state.saw_no_children = True
                state.final_nodes.append(node)
                state.best_confidence = _max_confidence(
                    state.best_confidence, step.confidence
                )
        if child_tasks:
            child_states = await asyncio.gather(*child_tasks)
            for child_state in child_states:
                state.path.extend(child_state.path[base_path_len:])
                state.path_nodes.extend(child_state.path_nodes[base_path_nodes_len:])
                state.final_nodes.extend(child_state.final_nodes[base_final_nodes_len:])
                state.best_confidence = _max_confidence(
                    state.best_confidence, child_state.best_confidence
                )
                state.saw_max_depth = state.saw_max_depth or child_state.saw_max_depth
                state.saw_no_children = (
                    state.saw_no_children or child_state.saw_no_children
                )
                state.saw_terminal_stop = (
                    state.saw_terminal_stop or child_state.saw_terminal_stop
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

    def _build_sub_agent(
        self,
        nodes: Sequence[TaxonomyNode],
    ) -> "TaxonomyClassifierAgent":
        """Build a classifier agent for a taxonomy subtree.

        Parameters
        ----------
        nodes : Sequence[TaxonomyNode]
            Taxonomy nodes to use as the sub-agent's root taxonomy.

        Returns
        -------
        TaxonomyClassifierAgent
            Configured classifier agent for the taxonomy slice.
        """
        return TaxonomyClassifierAgent(
            template_path=self._template_path,
            model=self._model,
            taxonomy=list(nodes),
        )

    async def _classify_subtree(
        self,
        *,
        sub_agent: "TaxonomyClassifierAgent",
        text: str,
        nodes: list[TaxonomyNode],
        depth: int,
        parent_path: list[str],
        context: Optional[Dict[str, Any]],
        max_depth: Optional[int],
        confidence_threshold: float | None,
        single_class: bool,
        state: "_TraversalState",
    ) -> "_TraversalState":
        """Classify a taxonomy subtree and return the traversal state.

        Parameters
        ----------
        sub_agent : TaxonomyClassifierAgent
            Sub-agent configured for the subtree traversal.
        text : str
            Source text to classify.
        nodes : list[TaxonomyNode]
            Candidate taxonomy nodes for the subtree.
        depth : int
            Current traversal depth.
        parent_path : list[str]
            Path segments leading to the current subtree.
        context : dict or None
            Additional context values to merge into the prompt.
        max_depth : int or None
            Maximum traversal depth before stopping.
        confidence_threshold : float or None
            Minimum confidence required to accept a classification step.
        single_class : bool
            Whether to keep only the highest-priority selection per step.
        state : _TraversalState
            Traversal state to populate for the subtree.

        Returns
        -------
        _TraversalState
            Populated traversal state for the subtree.
        """
        await sub_agent._classify_nodes(
            text=text,
            nodes=nodes,
            depth=depth,
            parent_path=parent_path,
            context=context,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            single_class=single_class,
            state=state,
        )
        return state


@dataclass
class _TraversalState:
    """Track recursive traversal state."""

    path: list[ClassificationStep] = field(default_factory=list)
    path_nodes: list[TaxonomyNode] = field(default_factory=list)
    final_nodes: list[TaxonomyNode] = field(default_factory=list)
    best_confidence: float | None = None
    saw_max_depth: bool = False
    saw_no_children: bool = False
    saw_terminal_stop: bool = False


def _copy_traversal_state(state: _TraversalState) -> _TraversalState:
    """Copy traversal state for parallel subtree execution.

    Parameters
    ----------
    state : _TraversalState
        Traversal state to clone.

    Returns
    -------
    _TraversalState
        Cloned traversal state with copied collections.
    """
    return _TraversalState(
        path=list(state.path),
        path_nodes=list(state.path_nodes),
        final_nodes=list(state.final_nodes),
        best_confidence=state.best_confidence,
        saw_max_depth=state.saw_max_depth,
        saw_no_children=state.saw_no_children,
        saw_terminal_stop=state.saw_terminal_stop,
    )


def _resolve_stop_reason(state: _TraversalState) -> ClassificationStopReason:
    """Resolve the final stop reason based on traversal state.

    Parameters
    ----------
    state : _TraversalState
        Traversal state to inspect.

    Returns
    -------
    ClassificationStopReason
        Resolved stop reason.
    """
    if state.saw_terminal_stop:
        return ClassificationStopReason.STOP
    if state.final_nodes and state.saw_no_children:
        return ClassificationStopReason.NO_CHILDREN
    if state.final_nodes:
        return ClassificationStopReason.STOP
    if state.saw_max_depth:
        return ClassificationStopReason.MAX_DEPTH
    if state.saw_no_children:
        return ClassificationStopReason.NO_CHILDREN
    return ClassificationStopReason.NO_MATCH


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
) -> type[ClassificationStep]:
    """Build a step output structure constrained to taxonomy paths.

    Parameters
    ----------
    path_identifiers : Sequence[str]
        Path identifiers for nodes at the current classification step.

    Returns
    -------
    type[ClassificationStep]
        Dynamic structure class for the classification step output.
    """
    node_enum = _build_taxonomy_enum("TaxonomyPath", path_identifiers)
    return ClassificationStep.build_for_enum(node_enum)


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
    seen: dict[str, int] = {}
    for node in nodes:
        base_path = _format_path_identifier([*parent_path, node.label])
        count = seen.get(base_path, 0) + 1
        seen[base_path] = count
        path = f"{base_path} ({count})" if count > 1 else base_path
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


def _format_path_identifier(path_segments: Sequence[str]) -> str:
    """Format path segments into a safe identifier string.

    Parameters
    ----------
    path_segments : Sequence[str]
        Path segments to format.

    Returns
    -------
    str
        Escaped path identifier string.
    """
    delimiter = " > "
    escape_token = "\\>"
    escaped_segments = [
        segment.replace(delimiter, escape_token) for segment in path_segments
    ]
    return delimiter.join(escaped_segments)


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
    normalized: dict[str, Any] = {}
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
    selected_nodes = _selected_nodes(step)
    if selected_nodes:
        for selected_node in selected_nodes:
            node = node_paths.get(selected_node)
            if node:
                resolved.append(node)
    return resolved


def _selected_nodes(step: ClassificationStep) -> list[str]:
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
    if step.selected_nodes is not None:
        selected_nodes = [
            selected_node for selected_node in step.selected_nodes if selected_node
        ]
        if selected_nodes:
            return selected_nodes
    return [step.selected_node] if step.selected_node else []


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

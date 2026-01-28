"""Agent for taxonomy-driven text classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from ..structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    TaxonomyNode,
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
    ...     TaxonomyNode(id="billing", label="Billing"),
    ...     TaxonomyNode(id="support", label="Support"),
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
        ...     id="finance",
        ...     label="Finance",
        ...     children=[TaxonomyNode(id="tax", label="Tax")],
        ... )
        >>> agent = TaxonomyClassifierAgent(model="gpt-4o-mini", taxonomy=taxonomy)
        >>> isinstance(agent.root_nodes, list)
        True
        """
        path: list[ClassificationStep] = []
        path_nodes: list[TaxonomyNode] = []
        stop_reason = ClassificationStopReason.NO_MATCH
        confidence = None
        branch_queue = [_BranchState(nodes=list(self._root_nodes), depth=0)]
        final_nodes: list[TaxonomyNode] = []

        while branch_queue:
            branch = branch_queue.pop(0)
            current_nodes = branch.nodes
            depth = branch.depth
            if max_depth is not None and depth >= max_depth:
                stop_reason = ClassificationStopReason.MAX_DEPTH
                continue
            if not current_nodes:
                continue

            template_context = _build_context(
                current_nodes=current_nodes,
                path=path,
                depth=depth,
                context=context,
            )
            step: ClassificationStep = await self.run_async(
                input=text,
                context=template_context,
                output_structure=ClassificationStep,
            )
            path.append(step)
            stop_reason = step.stop_reason
            confidence = step.confidence

            if (
                confidence_threshold is not None
                and step.confidence is not None
                and step.confidence < confidence_threshold
            ):
                stop_reason = ClassificationStopReason.NO_MATCH
                continue

            resolved_nodes = _resolve_nodes(current_nodes, step)
            if resolved_nodes:
                if single_class:
                    resolved_nodes = resolved_nodes[:1]
                path_nodes.extend(resolved_nodes)

            if step.stop_reason.is_terminal:
                if resolved_nodes:
                    final_nodes.extend(resolved_nodes)
                continue

            if not resolved_nodes:
                stop_reason = ClassificationStopReason.NO_MATCH
                continue

            for node in resolved_nodes:
                if node.children:
                    branch_queue.append(
                        _BranchState(nodes=list(node.children), depth=depth + 1)
                    )
                else:
                    stop_reason = ClassificationStopReason.NO_CHILDREN
                    final_nodes.append(node)

        final_nodes_value = final_nodes or None
        final_node = final_nodes[0] if final_nodes else None
        return ClassificationResult(
            final_node=final_node,
            final_nodes=final_nodes_value,
            confidence=confidence,
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
    current_nodes: Iterable[TaxonomyNode],
    path: Sequence[ClassificationStep],
    depth: int,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the template context for a classification step.

    Parameters
    ----------
    current_nodes : Iterable[TaxonomyNode]
        Nodes available at the current taxonomy level.
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
        "taxonomy_nodes": list(current_nodes),
        "path": [step.as_summary() for step in path],
        "depth": depth,
    }
    if context:
        template_context.update(context)
    return template_context


def _resolve_nodes(
    nodes: Sequence[TaxonomyNode],
    step: ClassificationStep,
) -> list[TaxonomyNode]:
    """Resolve selected taxonomy nodes for a classification step.

    Parameters
    ----------
    nodes : Sequence[TaxonomyNode]
        Candidate nodes at the current level.
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
            for node in nodes:
                if node.id == selected_id:
                    resolved.append(node)
        if resolved:
            return resolved
    selected_labels = _selected_labels(step)
    for selected_label in selected_labels:
        for node in nodes:
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


__all__ = ["TaxonomyClassifierAgent"]

"""Agent for taxonomy-driven text classification."""

from __future__ import annotations

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
        depth = 0
        stop_reason = ClassificationStopReason.NO_MATCH
        current_nodes = list(self._root_nodes)

        while current_nodes:
            if max_depth is not None and depth >= max_depth:
                stop_reason = ClassificationStopReason.MAX_DEPTH
                break

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

            if step.stop_reason.is_terminal:
                break

            selected_node = _resolve_node(current_nodes, step)
            if selected_node is None:
                stop_reason = ClassificationStopReason.NO_MATCH
                break
            if not selected_node.children:
                stop_reason = ClassificationStopReason.NO_CHILDREN
                break

            current_nodes = list(selected_node.children)
            depth += 1

        final_id, final_label, confidence, final_ids, final_labels = _final_values(path)
        return ClassificationResult(
            final_id=final_id,
            final_ids=final_ids,
            final_label=final_label,
            final_labels=final_labels,
            confidence=confidence,
            stop_reason=stop_reason,
            path=path,
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


def _resolve_node(
    nodes: Sequence[TaxonomyNode],
    step: ClassificationStep,
) -> Optional[TaxonomyNode]:
    """Resolve the selected node for a classification step.

    Parameters
    ----------
    nodes : Sequence[TaxonomyNode]
        Candidate nodes at the current level.
    step : ClassificationStep
        Classification step output to resolve.

    Returns
    -------
    TaxonomyNode or None
        Matching taxonomy node if found.
    """
    selected_ids = _selected_ids(step)
    for selected_id in selected_ids:
        for node in nodes:
            if node.id == selected_id:
                return node
    selected_labels = _selected_labels(step)
    for selected_label in selected_labels:
        for node in nodes:
            if node.label == selected_label:
                return node
    return None


def _final_values(
    path: Sequence[ClassificationStep],
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[float],
    list[str] | None,
    list[str] | None,
]:
    """Return the final selection values from the path.

    Parameters
    ----------
    path : Sequence[ClassificationStep]
        Recorded classification steps.

    Returns
    -------
    tuple[str or None, str or None, float or None, list[str] or None, list[str] or None]
        Final identifier, label, confidence, and multi-class selections.
    """
    if not path:
        return None, None, None, None, None
    last_step = path[-1]
    selected_ids = _selected_ids(last_step) or None
    selected_labels = _selected_labels(last_step) or None
    final_id = selected_ids[0] if selected_ids else last_step.selected_id
    final_label = selected_labels[0] if selected_labels else last_step.selected_label
    return (
        final_id,
        final_label,
        last_step.confidence,
        selected_ids,
        selected_labels,
    )


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
    if step.selected_ids:
        return [selected_id for selected_id in step.selected_ids if selected_id]
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
    if step.selected_labels:
        return [
            selected_label for selected_label in step.selected_labels if selected_label
        ]
    return [step.selected_label] if step.selected_label else []


__all__ = ["TaxonomyClassifierAgent"]

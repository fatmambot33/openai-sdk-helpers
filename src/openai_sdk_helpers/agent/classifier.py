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
    """

    def __init__(
        self,
        *,
        template_path: Path | str | None = None,
        model: str | None = None,
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
        taxonomy: TaxonomyNode | Sequence[TaxonomyNode],
        *,
        context: Optional[Dict[str, Any]] = None,
        max_depth: Optional[int] = None,
    ) -> ClassificationResult:
        """Classify ``text`` by iterating over taxonomy levels.

        Parameters
        ----------
        text : str
            Source text to classify.
        taxonomy : TaxonomyNode or Sequence[TaxonomyNode]
            Root taxonomy node or list of root nodes to traverse.
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
            If ``taxonomy`` is empty.
        """
        roots = _normalize_roots(taxonomy)
        if not roots:
            raise ValueError("taxonomy must include at least one node")

        path: list[ClassificationStep] = []
        depth = 0
        stop_reason = ClassificationStopReason.NO_MATCH
        current_nodes = list(roots)

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

        final_id, final_label, confidence = _final_values(path)
        return ClassificationResult(
            final_id=final_id,
            final_label=final_label,
            confidence=confidence,
            stop_reason=stop_reason,
            path=path,
        )


def _normalize_roots(
    taxonomy: TaxonomyNode | Sequence[TaxonomyNode],
) -> list[TaxonomyNode]:
    """Normalize taxonomy input into a list of root nodes.

    Parameters
    ----------
    taxonomy : TaxonomyNode or Sequence[TaxonomyNode]
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
    if step.selected_id:
        for node in nodes:
            if node.id == step.selected_id:
                return node
    if step.selected_label:
        for node in nodes:
            if node.label == step.selected_label:
                return node
    return None


def _final_values(
    path: Sequence[ClassificationStep],
) -> tuple[Optional[str], Optional[str], Optional[float]]:
    """Return the final selection values from the path.

    Parameters
    ----------
    path : Sequence[ClassificationStep]
        Recorded classification steps.

    Returns
    -------
    tuple[str or None, str or None, float or None]
        Final identifier, label, and confidence.
    """
    if not path:
        return None, None, None
    last_step = path[-1]
    return last_step.selected_id, last_step.selected_label, last_step.confidence


__all__ = ["TaxonomyClassifierAgent"]

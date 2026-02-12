"""Response-native taxonomy classification helpers."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, cast

from ..prompt import PromptRenderer
from ..settings import OpenAISettings
from ..structure import (
    ClassificationResult,
    ClassificationStep,
    ClassificationStopReason,
    ClassificationSummary,
    StructureBase,
    TaxonomyNode,
    format_path_identifier,
    split_path_identifier,
)

_CONTINUE_CONFIDENCE_THRESHOLD = 0.7


class TaxonomyClassifierResponse:
    """Classify content by recursively traversing a taxonomy.

    This class mirrors ``TaxonomyClassifierAgent`` traversal and output
    semantics while executing each classification step through the Responses API
    in structure-only mode.

    Parameters
    ----------
    taxonomy : list[TaxonomyNode] | type[Enum]
        Root taxonomy nodes or enum-backed taxonomy values.
    model : str
        Model identifier used for classification requests.
    temperature : float | None, default=0
        Sampling temperature for classification requests.
    return_summary : bool, default=False
        Return ``ClassificationSummary`` instead of ``ClassificationResult``.
    data_path : Path | None, default=None
        Optional path reserved for compatibility with response helpers.

    Methods
    -------
    run_async(content, max_depth, confidence_threshold)
        Classify text asynchronously using taxonomy traversal.
    run_sync(content, max_depth, confidence_threshold)
        Classify text synchronously using taxonomy traversal.
    """

    def __init__(
        self,
        *,
        taxonomy: list[TaxonomyNode] | type[Enum],
        model: str,
        temperature: float | None = 0,
        return_summary: bool = False,
        data_path: Path | None = None,
    ) -> None:
        """Initialize response-mode taxonomy classifier configuration.

        Parameters
        ----------
        taxonomy : list[TaxonomyNode] | type[Enum]
            Root taxonomy nodes or enum-backed taxonomy values.
        model : str
            Model identifier used for classification requests.
        temperature : float | None, default=0
            Sampling temperature for classification requests.
        return_summary : bool, default=False
            Return ``ClassificationSummary`` instead of ``ClassificationResult``.
        data_path : Path | None, default=None
            Optional path reserved for compatibility with response helpers.

        Raises
        ------
        ValueError
            If the taxonomy contains no root nodes.
        """
        self._root_nodes = _normalize_taxonomy_input(taxonomy)
        self._taxonomy = taxonomy
        self._model = model
        self._temperature = temperature
        self._return_summary = return_summary
        self._data_path = data_path
        self._renderer = PromptRenderer()
        self._client: Any | None = None

    async def run_async(
        self,
        content: str,
        *,
        max_depth: int | None = None,
        confidence_threshold: float = 0.6,
    ) -> ClassificationResult | ClassificationSummary:
        """Classify content asynchronously with taxonomy traversal.

        Parameters
        ----------
        content : str
            Source text to classify.
        max_depth : int | None, default=None
            Maximum depth to traverse before stopping.
        confidence_threshold : float, default=0.6
            Minimum confidence required to accept a classification step.

        Returns
        -------
        ClassificationResult or ClassificationSummary
            Full traversal output or summary output.
        """
        result = await self._run_response(
            content,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
        )
        return _finalize_result(result, return_summary=self._return_summary)

    def run_sync(
        self,
        content: str,
        *,
        max_depth: int | None = None,
        confidence_threshold: float = 0.6,
    ) -> ClassificationResult | ClassificationSummary:
        """Classify content synchronously with taxonomy traversal.

        Parameters
        ----------
        content : str
            Source text to classify.
        max_depth : int | None, default=None
            Maximum depth to traverse before stopping.
        confidence_threshold : float, default=0.6
            Minimum confidence required to accept a classification step.

        Returns
        -------
        ClassificationResult or ClassificationSummary
            Full traversal output or summary output.
        """

        async def runner() -> ClassificationResult | ClassificationSummary:
            return await self.run_async(
                content,
                max_depth=max_depth,
                confidence_threshold=confidence_threshold,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())

        result: ClassificationResult | ClassificationSummary | None = None
        error: Exception | None = None

        def _thread_func() -> None:
            nonlocal result, error
            try:
                result = asyncio.run(runner())
            except Exception as exc:  # pragma: no cover - defensive branch.
                error = exc

        thread = threading.Thread(target=_thread_func)
        thread.start()
        thread.join()

        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("Classification did not return a result")
        return result

    async def _run_response(
        self,
        text: str,
        *,
        max_depth: int | None,
        confidence_threshold: float | None,
    ) -> ClassificationResult:
        """Classify text by recursively walking taxonomy levels.

        Parameters
        ----------
        text : str
            Source text to classify.
        max_depth : int | None
            Maximum depth to traverse before stopping.
        confidence_threshold : float | None
            Minimum confidence required to accept a classification step.

        Returns
        -------
        ClassificationResult
            Structured classification result for traversal.
        """
        state = _TraversalState()
        await self._classify_nodes(
            content=text,
            nodes=list(self._root_nodes),
            depth=0,
            parent_path=[],
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            state=state,
        )

        final_nodes_value = state.final_nodes or None
        return ClassificationResult(
            final_nodes=final_nodes_value,
            confidence=state.best_confidence,
            stop_reason=_resolve_stop_reason(state),
            steps=state.steps,
        )

    async def _classify_nodes(
        self,
        *,
        content: str,
        nodes: list[TaxonomyNode],
        depth: int,
        parent_path: list[str],
        max_depth: int | None,
        confidence_threshold: float | None,
        state: "_TraversalState",
    ) -> None:
        """Classify one taxonomy level and recurse into selected nodes.

        Parameters
        ----------
        content : str
            Source text to classify.
        nodes : list[TaxonomyNode]
            Candidate taxonomy nodes at the current level.
        depth : int
            Current traversal depth.
        parent_path : list[str]
            Path segments leading to the current level.
        max_depth : int | None
            Maximum traversal depth before stopping.
        confidence_threshold : float | None
            Minimum confidence required to accept a classification step.
        state : _TraversalState
            Aggregated traversal state.
        """
        if max_depth is not None and depth >= max_depth:
            state.saw_max_depth = True
            return
        if not nodes:
            return

        node_paths = _build_node_path_map(nodes, parent_path)
        step_structure = _build_step_structure(list(node_paths.keys()))
        step_context = _build_context(
            node_descriptors=_build_node_descriptors(node_paths),
            steps=state.steps,
            depth=depth,
            context=None,
        )
        raw_step = await self._run_step_async(
            content=content,
            context=step_context,
            output_structure=step_structure,
        )
        step = _normalize_step_output(raw_step)
        state.steps.append(step)

        if (
            confidence_threshold is not None
            and step.confidence is not None
            and step.confidence < confidence_threshold
        ):
            return

        resolved_nodes = _resolve_nodes(node_paths, step)
        should_continue = _should_continue_from_stop(step, resolved_nodes)
        if should_continue:
            step = step.model_copy(
                update={"stop_reason": ClassificationStopReason.CONTINUE}
            )
            state.steps[-1] = step

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

        for node in resolved_nodes:
            if node.children:
                prior_final_count = len(state.final_nodes)
                await self._classify_nodes(
                    content=content,
                    nodes=node.children,
                    depth=depth + 1,
                    parent_path=[*parent_path, node.label],
                    max_depth=max_depth,
                    confidence_threshold=confidence_threshold,
                    state=state,
                )
                branch_final_nodes = state.final_nodes[prior_final_count:]
                if should_continue and not branch_final_nodes:
                    state.final_nodes.append(node)
                    state.best_confidence = _max_confidence(
                        state.best_confidence, step.confidence
                    )
            else:
                state.saw_no_children = True
                state.final_nodes.append(node)
                state.best_confidence = _max_confidence(
                    state.best_confidence, step.confidence
                )

    async def _run_step_async(
        self,
        *,
        content: str,
        context: dict[str, Any],
        output_structure: type[StructureBase],
    ) -> StructureBase:
        """Execute one classification step via Responses API.

        Parameters
        ----------
        content : str
            Source text to classify.
        context : dict[str, Any]
            Prompt-rendering context containing candidate taxonomy nodes.
        output_structure : type[StructureBase]
            Dynamic structure with enum-constrained path identifiers.

        Returns
        -------
        StructureBase
            Parsed step output for the current taxonomy level.

        Raises
        ------
        RuntimeError
            If the API returns no structured output.
        """
        prompt = self._renderer.render("classifier.jinja", context=context)
        payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_text", "text": f"Text to classify:\n{content}"},
                ],
            }
        ]
        response = await asyncio.to_thread(
            self._get_client().responses.create,
            model=self._model,
            input=payload,
            text=output_structure.response_format(),
            temperature=self._temperature,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RuntimeError("No structured output returned from Responses API")
        parsed = json.loads(output_text)
        return output_structure.from_json(parsed)

    def _get_client(self) -> Any:
        """Return a cached OpenAI client.

        Returns
        -------
        Any
            OpenAI client instance.
        """
        if self._client is not None:
            return self._client
        openai_settings = OpenAISettings.from_env(default_model=self._model)
        self._client = openai_settings.create_client()
        return self._client


@dataclass
class _TraversalState:
    """Track recursive traversal state."""

    steps: list[ClassificationStep] = field(default_factory=list)
    final_nodes: list[TaxonomyNode] = field(default_factory=list)
    best_confidence: float | None = None
    saw_max_depth: bool = False
    saw_no_children: bool = False
    saw_terminal_stop: bool = False


def classify_taxonomy_response(
    *,
    content: str,
    taxonomy: list[TaxonomyNode] | type[Enum],
    model: str,
    temperature: float | None = 0,
    max_depth: int | None = None,
    confidence_threshold: float = 0.6,
    return_summary: bool = False,
    data_path: Path | None = None,
) -> ClassificationResult | ClassificationSummary:
    """Classify text against one taxonomy via the Responses API.

    Parameters
    ----------
    content : str
        Source text to classify.
    taxonomy : list[TaxonomyNode] | type[Enum]
        Root taxonomy nodes or enum-backed taxonomy values.
    model : str
        Model identifier used for classification requests.
    temperature : float | None, default=0
        Sampling temperature for classification requests.
    max_depth : int | None, default=None
        Maximum depth to traverse before stopping.
    confidence_threshold : float, default=0.6
        Minimum confidence required to accept a classification step.
    return_summary : bool, default=False
        Return ``ClassificationSummary`` instead of ``ClassificationResult``.
    data_path : Path | None, default=None
        Optional path reserved for compatibility with response helpers.

    Returns
    -------
    ClassificationResult or ClassificationSummary
        Full traversal result or summary output.
    """
    classifier = TaxonomyClassifierResponse(
        taxonomy=taxonomy,
        model=model,
        temperature=temperature,
        return_summary=return_summary,
        data_path=data_path,
    )
    return classifier.run_sync(
        content,
        max_depth=max_depth,
        confidence_threshold=confidence_threshold,
    )


def _finalize_result(
    result: ClassificationResult,
    *,
    return_summary: bool,
) -> ClassificationResult | ClassificationSummary:
    """Return final classification output.

    Parameters
    ----------
    result : ClassificationResult
        Full traversal result.
    return_summary : bool
        Return summary output when True.

    Returns
    -------
    ClassificationResult or ClassificationSummary
        Full result or lightweight summary.
    """
    if not return_summary:
        return result
    summary = result.to_lightweight_summary()
    if summary is None:
        return ClassificationSummary(full_paths=[])
    full_paths = list(dict.fromkeys(summary.full_paths or []))
    return ClassificationSummary(full_paths=full_paths)


def _resolve_stop_reason(state: _TraversalState) -> ClassificationStopReason:
    """Resolve final stop reason from traversal state.

    Parameters
    ----------
    state : _TraversalState
        Traversal state to inspect.

    Returns
    -------
    ClassificationStopReason
        Final stop reason.
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


def _should_continue_from_stop(
    step: ClassificationStep,
    resolved_nodes: Sequence[TaxonomyNode],
    threshold: float = _CONTINUE_CONFIDENCE_THRESHOLD,
) -> bool:
    """Return True when stop reason should continue traversal.

    Parameters
    ----------
    step : ClassificationStep
        Classification step to evaluate.
    resolved_nodes : Sequence[TaxonomyNode]
        Resolved taxonomy nodes from the classification step.
    threshold : float, default=0.7
        Confidence threshold for overriding stop reason.

    Returns
    -------
    bool
        True when traversal should continue.
    """
    if step.stop_reason is not ClassificationStopReason.STOP:
        return False
    if step.confidence is None or step.confidence < threshold:
        return False
    return any(node.children for node in resolved_nodes)


def _normalize_taxonomy_input(
    taxonomy: list[TaxonomyNode] | type[Enum],
) -> list[TaxonomyNode]:
    """Normalize taxonomy roots from nodes or enum values.

    Parameters
    ----------
    taxonomy : list[TaxonomyNode] | type[Enum]
        Taxonomy roots or enum-backed path identifiers.

    Returns
    -------
    list[TaxonomyNode]
        Normalized root nodes.

    Raises
    ------
    ValueError
        If taxonomy contains no nodes.
    """
    if isinstance(taxonomy, type) and issubclass(taxonomy, Enum):
        roots = _taxonomy_nodes_from_enum(taxonomy)
    else:
        roots = [node for node in taxonomy if node is not None]
    if not roots:
        raise ValueError("taxonomy must include at least one node")
    return roots


def _taxonomy_nodes_from_enum(enum_cls: type[Enum]) -> list[TaxonomyNode]:
    """Build taxonomy nodes from enum path values.

    Parameters
    ----------
    enum_cls : type[Enum]
        Enum class with path-like member values.

    Returns
    -------
    list[TaxonomyNode]
        Root taxonomy nodes preserving enum declaration order.
    """
    roots: list[TaxonomyNode] = []
    for member in enum_cls:
        raw_value = member.value
        if raw_value is None:
            continue
        path = split_path_identifier(str(raw_value))
        if not path:
            continue
        _insert_taxonomy_path(roots, path)
    return roots


def _insert_taxonomy_path(roots: list[TaxonomyNode], path: Sequence[str]) -> None:
    """Insert a taxonomy path into root nodes.

    Parameters
    ----------
    roots : list[TaxonomyNode]
        Root nodes to mutate.
    path : Sequence[str]
        Path segments to insert.
    """
    current_nodes = roots
    for segment in path:
        node = next((item for item in current_nodes if item.label == segment), None)
        if node is None:
            node = TaxonomyNode(label=segment)
            current_nodes.append(node)
        current_nodes = node.children


def _build_context(
    *,
    node_descriptors: Iterable[dict[str, Any]],
    steps: Sequence[ClassificationStep],
    depth: int,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build template context for one classification step.

    Parameters
    ----------
    node_descriptors : Iterable[dict[str, Any]]
        Node descriptors available at this taxonomy level.
    steps : Sequence[ClassificationStep]
        Previously recorded traversal steps.
    depth : int
        Current traversal depth.
    context : dict[str, Any] or None
        Optional additional context values.

    Returns
    -------
    dict[str, Any]
        Context dictionary for prompt rendering.
    """
    summarized_steps = [
        step.as_summary()
        for step in steps
        if step.selected_nodes and any(node is not None for node in step.selected_nodes)
    ]
    template_context: Dict[str, Any] = {
        "taxonomy_nodes": list(node_descriptors),
        "steps": summarized_steps,
        "depth": depth,
    }
    if context:
        template_context.update(context)
    return template_context


def _build_step_structure(path_identifiers: Sequence[str]) -> type[ClassificationStep]:
    """Build a step output structure constrained to taxonomy paths.

    Parameters
    ----------
    path_identifiers : Sequence[str]
        Path identifiers available for this step.

    Returns
    -------
    type[ClassificationStep]
        Dynamic structure class for classification output.
    """
    node_enum = _build_taxonomy_enum("TaxonomyPath", path_identifiers)
    return ClassificationStep.build_for_enum(node_enum)


def _build_node_path_map(
    nodes: Sequence[TaxonomyNode],
    parent_path: Sequence[str],
) -> dict[str, TaxonomyNode]:
    """Build a mapping of path identifiers to taxonomy nodes.

    Parameters
    ----------
    nodes : Sequence[TaxonomyNode]
        Candidate nodes at the current level.
    parent_path : Sequence[str]
        Path segments leading to the current level.

    Returns
    -------
    dict[str, TaxonomyNode]
        Mapping from path identifiers to taxonomy nodes.
    """
    path_map: dict[str, TaxonomyNode] = {}
    seen: dict[str, int] = {}
    for node in nodes:
        base_path = format_path_identifier([*parent_path, node.label])
        count = seen.get(base_path, 0) + 1
        seen[base_path] = count
        path = f"{base_path} ({count})" if count > 1 else base_path
        path_map[path] = node
    return path_map


def _build_node_descriptors(
    node_paths: dict[str, TaxonomyNode],
) -> list[dict[str, Any]]:
    """Build taxonomy node descriptors for prompt rendering.

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
                "computed_description": node.computed_description,
            }
        )
    return descriptors


def _build_taxonomy_enum(name: str, values: Sequence[str]) -> type[Enum]:
    """Build a safe enum from taxonomy path values.

    Parameters
    ----------
    name : str
        Enum class name.
    values : Sequence[str]
        Taxonomy path values used as enum members.

    Returns
    -------
    type[Enum]
        Enum class with sanitized member names.
    """
    members: dict[str, str] = {}
    for index, value in enumerate(values, start=1):
        member_name = _sanitize_enum_member(value, index, members)
        members[member_name] = value
    if not members:
        members["UNSPECIFIED"] = ""
    return cast(type[Enum], Enum(name, members))


def _sanitize_enum_member(
    value: str,
    index: int,
    existing: dict[str, str],
) -> str:
    """Return a valid enum member name for taxonomy values.

    Parameters
    ----------
    value : str
        Raw taxonomy value to sanitize.
    index : int
        Value index in source sequence.
    existing : dict[str, str]
        Existing members to avoid naming collisions.

    Returns
    -------
    str
        Sanitized enum member name.
    """
    normalized_segments: list[str] = []
    for segment in split_path_identifier(value):
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", segment).strip("_").upper()
        if not normalized:
            normalized = "VALUE"
        if normalized[0].isdigit():
            normalized = f"VALUE_{normalized}"
        normalized_segments.append(normalized)
    normalized_path = "__".join(normalized_segments) or f"VALUE_{index}"
    candidate = normalized_path
    suffix = 1
    while candidate in existing:
        candidate = f"{normalized_path}__{suffix}"
        suffix += 1
    return candidate


def _normalize_step_output(step: StructureBase) -> ClassificationStep:
    """Normalize dynamic step output into ``ClassificationStep``.

    Parameters
    ----------
    step : StructureBase
        Raw step output returned by the API.

    Returns
    -------
    ClassificationStep
        Normalized classification step instance.
    """
    if isinstance(step, ClassificationStep):
        return step
    return ClassificationStep.from_json(step.to_json())


def _resolve_nodes(
    node_paths: dict[str, TaxonomyNode],
    step: ClassificationStep,
) -> list[TaxonomyNode]:
    """Resolve selected taxonomy nodes for one classification step.

    Parameters
    ----------
    node_paths : dict[str, TaxonomyNode]
        Mapping of path identifiers to nodes.
    step : ClassificationStep
        Step output to resolve.

    Returns
    -------
    list[TaxonomyNode]
        Matching taxonomy nodes in selected order.
    """
    resolved: list[TaxonomyNode] = []
    for selected_node in _selected_nodes(step):
        node = node_paths.get(selected_node)
        if node is not None:
            resolved.append(node)
    return resolved


def _selected_nodes(step: ClassificationStep) -> list[str]:
    """Return selected identifiers for a classification step.

    Parameters
    ----------
    step : ClassificationStep
        Classification step output.

    Returns
    -------
    list[str]
        Selected identifiers in priority order.
    """
    enum_cls: type[Enum] | None = None
    step_cls = step.__class__
    if hasattr(step_cls, "model_fields"):
        field = step_cls.model_fields.get("selected_nodes")
        if field is not None:
            enum_cls = step_cls._extract_enum_class(field.annotation)
    if enum_cls is None:
        enum_cls = Enum
    return [
        str(_normalize_enum_value(selected_node, enum_cls))
        for selected_node in step.selected_nodes or []
        if selected_node
    ]


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
        if value in enum_cls._value2member_map_:
            return enum_cls(value).value
        if value in enum_cls.__members__:
            return enum_cls.__members__[value].value
    return value


def _max_confidence(
    current: float | None,
    candidate: float | None,
) -> float | None:
    """Return the higher confidence value.

    Parameters
    ----------
    current : float | None
        Current best confidence value.
    candidate : float | None
        Candidate confidence value.

    Returns
    -------
    float | None
        Highest available confidence value.
    """
    if current is None:
        return candidate
    if candidate is None:
        return current
    return max(current, candidate)


__all__ = ["TaxonomyClassifierResponse", "classify_taxonomy_response"]

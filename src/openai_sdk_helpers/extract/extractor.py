"""Document extraction helpers powered by LangExtract."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..errors import ExtractionError, InputValidationError
from ..logging import log
from ..structure.extraction import ExtractionItem, ExtractionResult
from .schema import (
    SchemaLike,
    build_examples_from_schema,
    build_prompt_from_schema,
    validate_schema_dict,
)


class DocumentExtractor:
    """Extract structured data from documents using LangExtract.

    Parameters
    ----------
    model : str
        Model identifier to pass to LangExtract.
    schema : SchemaLike | None, optional
        Schema dictionary for building prompts and examples. Default is None.
    examples : Any | None, optional
        LangExtract examples payload. Default is None.
    prompt : str | None, optional
        Prompt string to override schema-derived prompts. Default is None.
    **defaults
        Default keyword arguments forwarded to LangExtract.

    Methods
    -------
    extract(text_or_docs, **kwargs)
        Extract structured data from one or more documents.
    aextract(text_or_docs, **kwargs)
        Asynchronously extract structured data using a thread.
    """

    def __init__(
        self,
        model: str,
        schema: SchemaLike | None = None,
        examples: Any | None = None,
        prompt: str | None = None,
        **defaults: Any,
    ) -> None:
        """Initialize the extractor."""
        self.model = model
        self.defaults = dict(defaults)
        self.schema: SchemaLike | None = None

        if schema is not None and isinstance(schema, Mapping):
            self.schema = validate_schema_dict(schema)
            if prompt is None:
                prompt = build_prompt_from_schema(self.schema)
            if examples is None:
                examples = build_examples_from_schema(self.schema)
        else:
            self.schema = schema

        self.prompt = prompt
        self.examples = examples

    def extract(
        self,
        text_or_docs: str | Sequence[Any],
        **kwargs: Any,
    ) -> ExtractionResult | list[ExtractionResult]:
        """Extract structured data from one or more documents.

        Parameters
        ----------
        text_or_docs : str | Sequence[Any]
            Document text or iterable of document payloads.
        **kwargs : Any
            Keyword arguments forwarded to LangExtract.

        Returns
        -------
        ExtractionResult | list[ExtractionResult]
            Extraction results for a single document or list of documents.

        Raises
        ------
        ExtractionError
            If LangExtract fails or output validation fails.
        InputValidationError
            If inputs are malformed.
        """
        documents = list(self._normalize_documents(text_or_docs))
        if not documents:
            raise InputValidationError("No documents provided for extraction.")

        return_partial = bool(kwargs.pop("return_partial", False))
        max_workers = kwargs.pop("max_workers", self.defaults.get("max_workers", 1))

        if len(documents) == 1:
            document_id, text = documents[0]
            return self._extract_single(
                document_id,
                text,
                langextract_kwargs=kwargs,
            )

        results: list[ExtractionResult | None] = [None] * len(documents)
        errors: list[dict[str, str]] = []
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map: dict[Any, tuple[int, str | None]] = {
                    executor.submit(
                        self._extract_single,
                        document_id,
                        text,
                        langextract_kwargs=kwargs,
                    ): (index, document_id)
                    for index, (document_id, text) in enumerate(documents)
                }
                for future in as_completed(future_map):
                    index, document_id = future_map[future]
                    try:
                        results[index] = future.result()
                    except ExtractionError as exc:
                        if return_partial:
                            errors.append({"document_id": document_id, "error": str(exc)})
                        else:
                            raise
        else:
            for index, (document_id, text) in enumerate(documents):
                try:
                    results[index] = self._extract_single(
                        document_id,
                        text,
                        langextract_kwargs=kwargs,
                    )
                except ExtractionError as exc:
                    if return_partial:
                        errors.append({"document_id": document_id, "error": str(exc)})
                    else:
                        raise

        ordered_results = [result for result in results if result is not None]

        if errors:
            for result in ordered_results:
                result.metrics.setdefault("errors", errors)
            log(f"Extraction completed with {len(errors)} error(s).")

        if ordered_results:
            total_items = sum(len(result.items) for result in ordered_results)
            log(
                "Batch extraction complete: "
                f"{len(ordered_results)} document(s), {total_items} item(s) total."
            )

        return ordered_results

    async def aextract(
        self,
        text_or_docs: str | Sequence[Any],
        **kwargs: Any,
    ) -> ExtractionResult | list[ExtractionResult]:
        """Asynchronously extract structured data using a thread.

        Parameters
        ----------
        text_or_docs : str | Sequence[Any]
            Document text or iterable of document payloads.
        **kwargs : Any
            Keyword arguments forwarded to LangExtract.

        Returns
        -------
        ExtractionResult | list[ExtractionResult]
            Extraction results for a single document or list of documents.
        """
        return await asyncio.to_thread(self.extract, text_or_docs, **kwargs)

    def _extract_single(
        self,
        document_id: str | None,
        text: str,
        *,
        langextract_kwargs: dict[str, Any],
    ) -> ExtractionResult:
        """Extract a single document using LangExtract."""
        start_time = time.perf_counter()
        extractor = _resolve_langextract_callable()
        kwargs = self._build_langextract_kwargs(langextract_kwargs)
        include_language = kwargs.pop("include_language", False)

        try:
            raw_output = extractor(text, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise ExtractionError(
                "LangExtract extraction failed.", context={"error": str(exc)}
            ) from exc

        items = _map_items(raw_output)
        metrics = _extract_metrics(raw_output)

        duration_s = time.perf_counter() - start_time
        metrics.update(
            {
                "duration_s": duration_s,
                "num_items": len(items),
                "density_per_1k_chars": _density_per_1k_chars(text, len(items)),
            }
        )
        if include_language:
            metrics["language"] = _detect_language(text)
        _log_pass_deltas(metrics)

        log(
            (
                "Extraction complete for document "
                f"{document_id or '<unknown>'}: {len(items)} items, "
                f"density={metrics['density_per_1k_chars']:.2f}/1k chars"
            )
        )

        return ExtractionResult(
            document_id=document_id,
            items=items,
            metrics=metrics,
        )

    def _build_langextract_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Build keyword arguments for LangExtract calls."""
        _ensure_provider_env()
        kwargs = dict(self.defaults)
        kwargs.update(overrides)

        if "model_id" not in kwargs:
            kwargs["model_id"] = self.model

        if self.prompt is not None and "prompt" not in kwargs:
            kwargs["prompt"] = self.prompt
        if self.examples is not None and "examples" not in kwargs:
            kwargs["examples"] = self.examples

        return kwargs

    @staticmethod
    def _normalize_documents(
        text_or_docs: str | Sequence[Any],
    ) -> Iterable[tuple[str | None, str]]:
        """Normalize document inputs into id/text pairs."""
        if isinstance(text_or_docs, str):
            yield None, text_or_docs
            return

        for entry in text_or_docs:
            if isinstance(entry, str):
                yield None, entry
            elif isinstance(entry, Mapping):
                text = entry.get("text")
                if not isinstance(text, str):
                    raise InputValidationError("Document mapping must include text.")
                document_id = entry.get("document_id") or entry.get("id")
                yield document_id, text
            elif isinstance(entry, Sequence) and len(entry) == 2:
                document_id, text = entry
                if not isinstance(text, str):
                    raise InputValidationError("Document tuple must include text string.")
                yield document_id if isinstance(document_id, str) else None, text
            else:
                raise InputValidationError(
                    "Documents must be strings, mappings with text, or (id, text) pairs."
                )


def run_extraction(
    text_or_docs: str | Sequence[Any],
    *,
    schema: SchemaLike | None = None,
    model: str,
    **kwargs: Any,
) -> ExtractionResult | list[ExtractionResult]:
    """Run a one-off extraction using LangExtract.

    Parameters
    ----------
    text_or_docs : str | Sequence[Any]
        Document text or iterable of document payloads.
    schema : SchemaLike | None, optional
        Schema dictionary for building prompts and examples. Default is None.
    model : str
        Model identifier to pass to LangExtract.
    **kwargs : Any
        Keyword arguments forwarded to LangExtract.

    Returns
    -------
    ExtractionResult | list[ExtractionResult]
        Extraction results for a single document or list of documents.
    """
    extractor = DocumentExtractor(model=model, schema=schema)
    return extractor.extract(text_or_docs, **kwargs)


def _resolve_langextract_callable() -> Any:
    """Resolve the LangExtract extract callable."""
    try:
        import importlib

        module = importlib.import_module("langextract")
    except ImportError as exc:
        raise ImportError(
            "LangExtract is required. Install with 'pip install openai-sdk-helpers[extract]'."
        ) from exc

    if hasattr(module, "extract"):
        return module.extract
    raise AttributeError("LangExtract module does not expose an extract function.")


def _map_items(raw_output: Any) -> list[ExtractionItem]:
    """Map raw LangExtract output into ExtractionItem models."""
    items_payload: Any
    if isinstance(raw_output, Mapping):
        if "items" in raw_output:
            items_payload = raw_output["items"]
        elif "extractions" in raw_output:
            items_payload = raw_output["extractions"]
        else:
            items_payload = raw_output
    else:
        items_payload = raw_output

    if isinstance(items_payload, Mapping):
        items_payload = [items_payload]

    if not isinstance(items_payload, Iterable):
        raise ExtractionError("LangExtract output does not contain iterable items.")

    items: list[ExtractionItem] = []
    for item in items_payload:
        if isinstance(item, ExtractionItem):
            items.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ExtractionError("LangExtract output item must be a mapping.")

        extraction_class = (
            item.get("extraction_class")
            or item.get("class")
            or item.get("label")
            or "unknown"
        )
        extraction_text = (
            item.get("extraction_text")
            or item.get("text")
            or item.get("value")
        )
        if not extraction_text:
            raise ExtractionError("LangExtract output item missing extraction_text.")

        source_span = _normalize_span(
            item.get("source_span")
            or item.get("span")
            or item.get("offsets")
            or item.get("source_offsets")
        )

        items.append(
            ExtractionItem(
                extraction_class=str(extraction_class),
                extraction_text=str(extraction_text),
                attributes=dict(item.get("attributes") or {}),
                source_span=source_span,
                source_id=item.get("source_id") or item.get("document_id"),
            )
        )

    return items


def _normalize_span(span: Any) -> tuple[int, int] | None:
    """Normalize span data into a tuple."""
    if span is None:
        return None
    if isinstance(span, tuple) and len(span) == 2:
        return int(span[0]), int(span[1])
    if isinstance(span, list) and len(span) == 2:
        return int(span[0]), int(span[1])
    if isinstance(span, Mapping) and "start" in span and "end" in span:
        return int(span["start"]), int(span["end"])
    return None


def _extract_metrics(raw_output: Any) -> dict[str, Any]:
    """Extract metrics payload from LangExtract output."""
    if isinstance(raw_output, Mapping):
        metrics = raw_output.get("metrics")
        if isinstance(metrics, Mapping):
            return dict(metrics)
    return {}


def _density_per_1k_chars(text: str, num_items: int) -> float:
    """Compute extraction density per 1000 characters."""
    if not text:
        return 0.0
    return (num_items / max(len(text), 1)) * 1000


def _detect_language(text: str) -> str:
    """Detect the document language when enabled."""
    try:
        from langdetect import detect
    except ImportError:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def _log_pass_deltas(metrics: dict[str, Any]) -> None:
    """Log per-pass deltas when multi-pass metrics are available."""
    pass_metrics = metrics.get("pass_metrics")
    if not isinstance(pass_metrics, list) or not pass_metrics:
        return

    deltas: list[str] = []
    previous = None
    for entry in pass_metrics:
        if not isinstance(entry, Mapping):
            continue
        num_items = entry.get("num_items")
        if not isinstance(num_items, int):
            continue
        if previous is None:
            deltas.append(f"pass1={num_items}")
        else:
            deltas.append(f"+{num_items - previous}")
        previous = num_items

    if deltas:
        log(f"Extraction pass deltas: {', '.join(deltas)}")


def _ensure_provider_env() -> None:
    """Ensure provider environment variables are configured."""
    env = os.environ
    if "LANGEXTRACT_OPENAI_API_KEY" not in env and "OPENAI_API_KEY" in env:
        env["LANGEXTRACT_OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    if "LANGEXTRACT_GOOGLE_API_KEY" not in env:
        if "GOOGLE_API_KEY" in env:
            env["LANGEXTRACT_GOOGLE_API_KEY"] = env["GOOGLE_API_KEY"]
        elif "GEMINI_API_KEY" in env:
            env["LANGEXTRACT_GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]


__all__ = ["DocumentExtractor", "ExtractionError", "run_extraction"]

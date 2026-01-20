"""Document extraction helpers powered by LangExtract."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import typing
import langextract as lx
from langextract.core.data import AnnotatedDocument
import textwrap
from openai_sdk_helpers import prompt

from ..errors import ExtractionError, InputValidationError
from ..logging import log
from ..structure.extraction import ExtractionItem, ExtractionResult


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
        prompt_description: str,
        examples: typing.Sequence[typing.Any],
        model_id: str,
        max_workers: int = 1,
    ) -> None:
        """Initialize the extractor."""
        self._model_id = model_id
        self.prompt = prompt_description
        self.examples = examples
        self.max_workers = max_workers

    def extract(self, input_text: str) -> list[ExtractionItem]:
        """Run the extraction."""
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=self.prompt,
            examples=self.examples,
            model_id=self._model_id,  # Automatically selects OpenAI provider
            api_key=os.environ.get("OPENAI_API_KEY"),
            fence_output=True,
            use_schema_constraints=False,
        )
        if isinstance(result, list):
            extracted_items = []
            for doc in result:
                extraction = ExtractionResult.from_annotated_document(doc)
                extracted_items.append(extraction)
            return extracted_items

        return [ExtractionItem.from_annotated_document(result)]


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


__all__ = ["DocumentExtractor", "ExtractionError"]

"""Tests for basic extraction helpers."""

from __future__ import annotations

import sys
import types
from typing import Any

from openai_sdk_helpers.extract import (
    DocumentExtractor,
    build_examples_from_schema,
    build_prompt_from_schema,
)


def test_schema_prompt_and_examples() -> None:
    """Build prompt and examples from a schema dictionary."""
    schema = {
        "name": "Invoice",
        "description": "Extract invoice data from the document.",
        "fields": [
            {
                "name": "invoice_number",
                "description": "Invoice identifier.",
                "type": "string",
                "required": True,
                "example": "INV-001",
            },
            {
                "name": "total_amount",
                "description": "Total due amount.",
                "type": "number",
            },
        ],
    }

    prompt = build_prompt_from_schema(schema)
    examples = build_examples_from_schema(schema)

    assert "Invoice:" in prompt
    assert "invoice_number (string) [required]" in prompt
    assert examples == [{"invoice_number": "INV-001"}]


def test_extract_maps_output(monkeypatch: Any) -> None:
    """Map LangExtract output into structured extraction results."""
    module = types.SimpleNamespace(
        extract=lambda text, **kwargs: {
            "items": [
                {
                    "extraction_class": "company",
                    "extraction_text": "OpenAI",
                    "attributes": {"role": "vendor"},
                    "source_span": [0, 6],
                }
            ],
            "metrics": {"passes": 1},
        }
    )
    monkeypatch.setitem(sys.modules, "langextract", module)

    extractor = DocumentExtractor(model="gpt-4o-mini")
    result = extractor.extract("OpenAI builds models.")

    assert result.items[0].extraction_class == "company"
    assert result.items[0].source_span == (0, 6)
    assert result.metrics["num_items"] == 1

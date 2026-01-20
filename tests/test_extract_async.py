"""Async extraction tests."""

from __future__ import annotations

import sys
import types

import pytest

from openai_sdk_helpers.extract import DocumentExtractor


@pytest.mark.asyncio
async def test_async_extraction_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure async extraction mirrors sync behavior."""
    module = types.SimpleNamespace(
        extract=lambda text, **kwargs: {
            "items": [
                {
                    "extraction_class": "company",
                    "extraction_text": "OpenAI",
                    "source_span": [0, 6],
                }
            ]
        }
    )
    monkeypatch.setitem(sys.modules, "langextract", module)

    extractor = DocumentExtractor(model="gpt-4o-mini")
    sync_result = extractor.extract("OpenAI builds models.")
    async_result = await extractor.aextract("OpenAI builds models.")

    assert sync_result.items[0].extraction_text == async_result.items[0].extraction_text

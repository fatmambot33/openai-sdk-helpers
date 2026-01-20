"""Error handling tests for extraction."""

from __future__ import annotations

import sys
import types

import pytest

from openai_sdk_helpers.errors import ExtractionError
from openai_sdk_helpers.extract import DocumentExtractor


def test_schema_validation_error() -> None:
    """Raise a clear error when schema input is invalid."""
    with pytest.raises(ValueError, match="fields"):
        DocumentExtractor(model="gpt-4o-mini", schema={"fields": []})


def test_langextract_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ImportError when LangExtract is unavailable."""
    monkeypatch.delitem(sys.modules, "langextract", raising=False)
    extractor = DocumentExtractor(model="gpt-4o-mini")

    with pytest.raises(ImportError, match="LangExtract is required"):
        extractor.extract("Sample text")


def test_langextract_failure_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap LangExtract failures in ExtractionError."""
    def _raise_error(text: str, **kwargs: object) -> None:
        raise ValueError("boom")

    module = types.SimpleNamespace(extract=_raise_error)
    monkeypatch.setitem(sys.modules, "langextract", module)
    extractor = DocumentExtractor(model="gpt-4o-mini")

    with pytest.raises(ExtractionError, match="LangExtract extraction failed"):
        extractor.extract("Sample text")

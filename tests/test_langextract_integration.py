"""Tests for LangExtract adapter utilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from openai_sdk_helpers.utils.langextract import (
    LangExtractAdapter,
    build_langextract_adapter,
)


class SampleModel(BaseModel):
    """Sample Pydantic model for LangExtract validation."""

    name: str
    count: int


class DummyExtractor:
    """Dummy extractor with an extract method for testing."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def extract(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """Return preset payload for testing."""
        return {"name": text, **self.payload}


def test_langextract_adapter_with_extract_method() -> None:
    """Use adapter with an object exposing extract()."""
    extractor = DummyExtractor({"count": 2})
    adapter = LangExtractAdapter(extractor=extractor)

    result = adapter.extract("alpha")

    assert result == {"name": "alpha", "count": 2}


def test_langextract_adapter_with_callable() -> None:
    """Use adapter with a callable extractor."""

    def extractor(text: str, **kwargs: Any) -> dict[str, Any]:
        return {"name": text, "count": 1}

    adapter = LangExtractAdapter(extractor=extractor)

    result = adapter.extract("beta")

    assert result == {"name": "beta", "count": 1}


def test_langextract_adapter_extract_to_model() -> None:
    """Validate extracted data into a Pydantic model."""

    def extractor(text: str, **kwargs: Any) -> dict[str, Any]:
        return {"name": text, "count": 3}

    adapter = LangExtractAdapter(extractor=extractor)

    model = adapter.extract_to_model("gamma", SampleModel)

    assert model == SampleModel(name="gamma", count=3)


def test_build_langextract_adapter_with_explicit_extractor() -> None:
    """Build adapter from explicit extractor without importing LangExtract."""

    def extractor(text: str, **kwargs: Any) -> dict[str, Any]:
        return {"name": text, "count": 4}

    adapter = build_langextract_adapter(extractor=extractor)

    result = adapter.extract("delta")

    assert result == {"name": "delta", "count": 4}

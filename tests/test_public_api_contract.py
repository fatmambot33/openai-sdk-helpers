"""Regression tests for the supported package-root API."""

import openai_sdk_helpers


def test_public_api_exports_are_unique() -> None:
    """Keep the supported root API deterministic."""
    assert len(openai_sdk_helpers.__all__) == len(set(openai_sdk_helpers.__all__))


def test_public_api_exports_are_importable() -> None:
    """Ensure each supported package-root symbol resolves."""
    for name in openai_sdk_helpers.__all__:
        assert getattr(openai_sdk_helpers, name) is not None

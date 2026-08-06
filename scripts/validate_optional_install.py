"""Validate clean installation profiles for optional package capabilities."""

from __future__ import annotations

import argparse
import importlib.util
from typing import Sequence


def _require_missing(module_name: str) -> None:
    """Assert that a module is absent from the current environment.

    Parameters
    ----------
    module_name : str
        Top-level module expected to be unavailable.
    """
    if importlib.util.find_spec(module_name) is not None:
        raise AssertionError(f"{module_name!r} must not be installed in core profile")


def _validate_core() -> None:
    """Validate the dependency-minimal base installation."""
    import openai_sdk_helpers
    import openai_sdk_helpers.agent
    import openai_sdk_helpers.structure

    _require_missing("langextract")
    _require_missing("streamlit")

    for name in ("DocumentExtractor", "ExtractorAgent", "DocumentStructure"):
        try:
            getattr(openai_sdk_helpers, name)
        except ImportError as exc:
            expected = 'pip install "openai-sdk-helpers[extract]"'
            if expected not in str(exc):
                raise AssertionError(
                    f"Missing-extra error for {name} did not include {expected!r}"
                ) from exc
        else:
            raise AssertionError(f"{name} unexpectedly loaded without extract extra")


def _validate_extract() -> None:
    """Validate the document extraction installation profile."""
    import langextract  # noqa: F401
    from openai_sdk_helpers import DocumentExtractor, DocumentStructure, ExtractorAgent
    from openai_sdk_helpers.extract import generate_document_extractor_config

    assert DocumentExtractor is not None
    assert DocumentStructure is not None
    assert ExtractorAgent is not None
    assert callable(generate_document_extractor_config)


def _validate_ui() -> None:
    """Validate the Streamlit UI installation profile."""
    import streamlit  # noqa: F401
    from openai_sdk_helpers.streamlit_app import (
        StreamlitAppConfig,
        StreamlitAppRegistry,
    )

    assert StreamlitAppConfig is not None
    assert StreamlitAppRegistry is not None


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation for one installation profile.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Command-line arguments. Uses ``sys.argv`` when omitted.

    Returns
    -------
    int
        Zero when the selected installation profile is valid.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("core", "extract", "ui", "all"))
    args = parser.parse_args(argv)

    if args.profile == "core":
        _validate_core()
    elif args.profile == "extract":
        _validate_extract()
    elif args.profile == "ui":
        _validate_ui()
    else:
        _validate_extract()
        _validate_ui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

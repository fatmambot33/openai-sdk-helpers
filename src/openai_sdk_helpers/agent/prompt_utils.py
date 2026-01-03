"""Shared helpers for locating bundled prompt templates.

Attributes
----------
DEFAULT_PROMPT_DIR : Path
    Default directory containing bundled prompt templates.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt"

__all__ = ["DEFAULT_PROMPT_DIR"]

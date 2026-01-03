"""Prompt rendering utilities for template-based text generation.

This module provides Jinja2-based template rendering functionality for
creating dynamic prompts with variable substitution and template inheritance.

Classes
-------
PromptRenderer
    Jinja2-based template renderer for dynamic prompt generation.
"""

from __future__ import annotations

from .base import PromptRenderer

__all__ = ["PromptRenderer"]

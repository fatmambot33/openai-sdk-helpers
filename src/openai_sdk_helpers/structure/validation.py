"""Structures describing guardrail validation results."""

from __future__ import annotations

from typing import List, Optional

from .base import BaseStructure, spec_field


class ValidationResultStructure(BaseStructure):
    """Capture guardrail validation findings for user and agent messages.

    Methods
    -------
    print()
        Return a formatted string representation of the stored fields.
    """

    input_safe: bool = spec_field(
        "input_safe",
        allow_null=False,
        description="Whether the user-provided input is allowed within the guardrails.",
    )
    output_safe: bool = spec_field(
        "output_safe",
        allow_null=False,
        description="Whether the agent output adheres to the safety guardrails.",
    )
    violations: List[str] = spec_field(
        "violations",
        allow_null=False,
        default_factory=list,
        description="Detected policy or safety issues that require mitigation.",
    )
    recommended_actions: List[str] = spec_field(
        "recommended_actions",
        allow_null=False,
        default_factory=list,
        description="Steps to remediate or respond to any detected violations.",
    )
    sanitized_output: Optional[str] = spec_field(
        "sanitized_output",
        description="Optional redacted or rewritten text that fits the guardrails.",
    )


__all__ = ["ValidationResultStructure"]

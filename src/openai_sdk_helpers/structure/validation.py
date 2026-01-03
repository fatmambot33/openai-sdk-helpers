"""Structures describing guardrail validation results.

This module defines Pydantic models for representing the results of guardrail
validation checks on user inputs and agent outputs.
"""

from __future__ import annotations

from .base import BaseStructure, spec_field


class ValidationResultStructure(BaseStructure):
    """Capture guardrail validation findings for user and agent messages.

    Represents the results of safety and policy validation checks performed
    on both user inputs and agent outputs, including detected violations
    and recommended remediation actions.

    Attributes
    ----------
    input_safe : bool
        Whether the user-provided input is allowed within the guardrails.
    output_safe : bool
        Whether the agent output adheres to the safety guardrails.
    violations : list[str]
        Detected policy or safety issues that require mitigation.
    recommended_actions : list[str]
        Steps to remediate or respond to any detected violations.
    sanitized_output : str or None
        Optional redacted or rewritten text that fits the guardrails.

    Methods
    -------
    print()
        Return a formatted string representation of the stored fields.

    Examples
    --------
    >>> result = ValidationResultStructure(
    ...     input_safe=True,
    ...     output_safe=True,
    ...     violations=[],
    ...     recommended_actions=[]
    ... )
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
    violations: list[str] = spec_field(
        "violations",
        allow_null=False,
        default_factory=list,
        description="Detected policy or safety issues that require mitigation.",
    )
    recommended_actions: list[str] = spec_field(
        "recommended_actions",
        allow_null=False,
        default_factory=list,
        description="Steps to remediate or respond to any detected violations.",
    )
    sanitized_output: str | None = spec_field(
        "sanitized_output",
        description="Optional redacted or rewritten text that fits the guardrails.",
    )


__all__ = ["ValidationResultStructure"]

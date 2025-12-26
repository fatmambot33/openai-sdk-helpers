"""Shared structured output models for summaries."""

from __future__ import annotations

from typing import List

from .base import BaseStructure, spec_field


class SummaryTopic(BaseStructure):
    """Capture a topic-level summary with supporting citations.

    Methods
    -------
    print()
        Return a formatted string representation of the stored fields.
    """

    topic: str = spec_field(
        "topic",
        default=...,
        description="Topic or micro-trend identified in the provided excerpts.",
    )
    summary: str = spec_field(
        "summary",
        default=...,
        description="Concise explanation of what the excerpts convey about the topic.",
    )
    citations: List[str] = spec_field(
        "citations",
        default_factory=list,
        description="Indices or short quotes that justify the topic summary.",
    )


class SummaryStructure(BaseStructure):
    """Defines the consolidated summary returned by the summarizer agent.

    Methods
    -------
    print()
        Return a formatted string representation of the stored fields.
    """

    text: str = spec_field(
        "text",
        default=...,
        description="Combined summary synthesized from the supplied excerpts.",
    )


class ExtendedSummaryStructure(SummaryStructure):
    """Extend ``SummaryStructure`` with optional topic breakdown metadata.

    Methods
    -------
    print()
        Return a formatted string representation of the stored fields.
    """

    metadata: List[SummaryTopic] = spec_field(
        "metadata",
        default_factory=list,
        description="Optional topic-level summaries with supporting citations.",
    )

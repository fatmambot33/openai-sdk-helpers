"""Structured output models for summaries.

This module defines Pydantic models for representing summarization results,
including topic-level summaries with citations and consolidated text summaries.
"""

from __future__ import annotations

from .base import BaseStructure, spec_field


class SummaryTopic(BaseStructure):
    """Capture a topic-level summary with supporting citations.

    Represents a single topic or micro-trend identified in source excerpts,
    along with a summary and supporting citations.

    Attributes
    ----------
    topic : str
        Topic or micro-trend identified in the provided excerpts.
    summary : str
        Concise explanation of what the excerpts convey about the topic.
    citations : list[str]
        Indices or short quotes that justify the topic summary.

    Methods
    -------
    print()
        Return a formatted string representation of the stored fields.

    Examples
    --------
    >>> topic = SummaryTopic(
    ...     topic="AI Trends",
    ...     summary="Growing adoption of AI",
    ...     citations=["Source 1", "Source 2"]
    ... )
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
    citations: list[str] = spec_field(
        "citations",
        default_factory=list,
        description="Indices or short quotes that justify the topic summary.",
    )


class SummaryStructure(BaseStructure):
    """Consolidated summary returned by the summarizer agent.

    Represents a synthesized summary text derived from multiple source excerpts.

    Attributes
    ----------
    text : str
        Combined summary synthesized from the supplied excerpts.

    Methods
    -------
    print()
        Return a formatted string representation of the stored fields.

    Examples
    --------
    >>> summary = SummaryStructure(text="This is a summary")
    """

    text: str = spec_field(
        "text",
        default=...,
        description="Combined summary synthesized from the supplied excerpts.",
    )


class ExtendedSummaryStructure(SummaryStructure):
    """Extended summary with optional topic breakdown metadata.

    Extends SummaryStructure to include topic-level summaries with citations,
    providing more granular insight into the summarization.

    Attributes
    ----------
    metadata : list[SummaryTopic]
        Optional topic-level summaries with supporting citations.

    Methods
    -------
    print()
        Return a formatted string representation of the stored fields.

    Examples
    --------
    >>> extended = ExtendedSummaryStructure(
    ...     text="Overall summary",
    ...     metadata=[SummaryTopic(topic="T1", summary="S1", citations=[])]
    ... )
    """

    metadata: list[SummaryTopic] = spec_field(
        "metadata",
        default_factory=list,
        description="Optional topic-level summaries with supporting citations.",
    )

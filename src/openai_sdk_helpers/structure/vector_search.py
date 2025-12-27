"""Shared structured output models for vector search."""

from __future__ import annotations

from typing import List

from .base import BaseStructure, spec_field


class VectorSearchItemStructure(BaseStructure):
    """A single vector search to perform."""

    reason: str = spec_field("reason")
    query: str = spec_field("query")


class VectorSearchPlanStructure(BaseStructure):
    """Collection of vector searches required to satisfy the query."""

    searches: List[VectorSearchItemStructure] = spec_field("searches")


class VectorSearchItemResultStructure(BaseStructure):
    """Result of a single vector search."""

    texts: List[str] = spec_field("texts")


class VectorSearchItemResultsStructure(BaseStructure):
    """Collection of search results returned from multiple queries.

    Failed searches are recorded in ``errors`` to allow callers to inspect
    partial outcomes without losing visibility into issues.

    Methods
    -------
    append(item)
        Add a search result to the collection.
    """

    item_results: List[VectorSearchItemResultStructure] = spec_field(
        "item_results", default_factory=list
    )
    errors: List[str] = spec_field("errors", default_factory=list)

    def append(self, item: VectorSearchItemResultStructure) -> None:
        """Add a search result to the collection.

        Parameters
        ----------
        item : VectorSearchItemResultStructure
            Result item to append.

        Returns
        -------
        None
        """
        self.item_results.append(item)


class VectorSearchReportStructure(BaseStructure):
    """Structured output from the vector search writer agent."""

    short_summary: str = spec_field("short_summary")
    markdown_report: str = spec_field("markdown_report")
    follow_up_questions: List[str] = spec_field("follow_up_questions")
    sources: List[str] = spec_field("sources")


class VectorSearchStructure(BaseStructure):
    """Complete output of a vector search workflow."""

    query: str = spec_field("query")
    plan: VectorSearchPlanStructure = spec_field("plan")
    results: VectorSearchItemResultsStructure = spec_field("results")
    report: VectorSearchReportStructure = spec_field("report")


__all__ = [
    "VectorSearchReportStructure",
    "VectorSearchItemStructure",
    "VectorSearchItemResultStructure",
    "VectorSearchItemResultsStructure",
    "VectorSearchPlanStructure",
    "VectorSearchStructure",
]

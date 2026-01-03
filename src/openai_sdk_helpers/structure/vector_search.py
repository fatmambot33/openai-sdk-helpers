"""Structured output models for vector search workflows.

This module defines Pydantic models for representing vector search plans,
results, and reports. These structures support multi-query vector search
workflows with error tracking and result aggregation.
"""

from __future__ import annotations

from .base import BaseStructure, spec_field


class VectorSearchItemStructure(BaseStructure):
    """A single vector search to perform.

    Represents one vector search query with rationale for its inclusion
    in a multi-query search plan.

    Attributes
    ----------
    reason : str
        Explanation for why this search is needed.
    query : str
        Vector search query text.

    Examples
    --------
    >>> item = VectorSearchItemStructure(
    ...     reason="Find related documents",
    ...     query="machine learning trends"
    ... )
    """

    reason: str = spec_field("reason")
    query: str = spec_field("query")


class VectorSearchPlanStructure(BaseStructure):
    """Collection of vector searches required to satisfy the query.

    Represents a plan containing multiple vector searches that together
    provide comprehensive coverage for answering a user query.

    Attributes
    ----------
    searches : list[VectorSearchItemStructure]
        List of vector search queries to execute.

    Examples
    --------
    >>> plan = VectorSearchPlanStructure(
    ...     searches=[VectorSearchItemStructure(reason="R", query="Q")]
    ... )
    """

    searches: list[VectorSearchItemStructure] = spec_field("searches")


class VectorSearchItemResultStructure(BaseStructure):
    """Result of a single vector search.

    Contains the text results retrieved from executing one vector search query.

    Attributes
    ----------
    texts : list[str]
        Retrieved text passages from the vector search.

    Examples
    --------
    >>> result = VectorSearchItemResultStructure(texts=["Result 1", "Result 2"])
    """

    texts: list[str] = spec_field("texts")


class VectorSearchItemResultsStructure(BaseStructure):
    """Collection of search results from multiple queries.

    Aggregates results from multiple vector searches while tracking any
    errors encountered. Failed searches are recorded in the errors list
    to allow inspection of partial outcomes.

    Attributes
    ----------
    item_results : list[VectorSearchItemResultStructure]
        List of successful search results.
    errors : list[str]
        List of error messages from failed searches.

    Methods
    -------
    append(item)
        Add a search result to the collection.

    Examples
    --------
    >>> results = VectorSearchItemResultsStructure()
    >>> results.append(VectorSearchItemResultStructure(texts=["Text"]))
    """

    item_results: list[VectorSearchItemResultStructure] = spec_field(
        "item_results", default_factory=list
    )
    errors: list[str] = spec_field("errors", default_factory=list)

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
    """Structured output from the vector search writer agent.

    Contains the final synthesized report from vector search results,
    including summary, markdown report, follow-up questions, and sources.

    Attributes
    ----------
    short_summary : str
        Brief summary of the search findings.
    markdown_report : str
        Full markdown-formatted report.
    follow_up_questions : list[str]
        Suggested questions for further exploration.
    sources : list[str]
        Source references used in the report.

    Examples
    --------
    >>> report = VectorSearchReportStructure(
    ...     short_summary="Summary",
    ...     markdown_report="# Report",
    ...     follow_up_questions=["Q1?"],
    ...     sources=["Source 1"]
    ... )
    """

    short_summary: str = spec_field("short_summary")
    markdown_report: str = spec_field("markdown_report")
    follow_up_questions: list[str] = spec_field("follow_up_questions")
    sources: list[str] = spec_field("sources")


class VectorSearchStructure(BaseStructure):
    """Complete output of a vector search workflow.

    Represents the full lifecycle of a vector search operation, from the
    original query through plan generation, execution, and final report.

    Attributes
    ----------
    query : str
        Original user query.
    plan : VectorSearchPlanStructure
        Generated search plan.
    results : VectorSearchItemResultsStructure
        Aggregated search results.
    report : VectorSearchReportStructure
        Final synthesized report.

    Examples
    --------
    >>> workflow = VectorSearchStructure(
    ...     query="Test query",
    ...     plan=VectorSearchPlanStructure(searches=[]),
    ...     results=VectorSearchItemResultsStructure(),
    ...     report=VectorSearchReportStructure(
    ...         short_summary="S",
    ...         markdown_report="R",
    ...         follow_up_questions=[],
    ...         sources=[]
    ...     )
    ... )
    """

    query: str = spec_field("query")
    plan: VectorSearchPlanStructure = spec_field("plan")
    results: VectorSearchItemResultsStructure = spec_field("results")
    report: VectorSearchReportStructure = spec_field("report")


__all__ = [
    "VectorSearchReportStructure",
    "VectorSearchItemStructure",
    "VectorSearchPlanStructure",
    "VectorSearchItemResultStructure",
    "VectorSearchItemResultsStructure",
    "VectorSearchStructure",
]

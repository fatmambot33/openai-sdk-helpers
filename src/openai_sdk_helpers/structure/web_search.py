"""Structured output models for web search workflows.

This module defines Pydantic models for representing web search plans,
results, and reports. These structures support multi-query web search
workflows with comprehensive reporting.
"""

from __future__ import annotations

from .base import StructureBase, spec_field


class WebSearchReportStructure(StructureBase):
    """Structured output from the web search writer agent.

    Contains the final synthesized report from web search results,
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
        Source URLs and references used in the report.

    Examples
    --------
    >>> report = WebSearchReportStructure(
    ...     short_summary="Summary",
    ...     markdown_report="# Report",
    ...     follow_up_questions=["Q1?"],
    ...     sources=["https://example.com"]
    ... )
    """

    short_summary: str = spec_field("short_summary")
    markdown_report: str = spec_field("markdown_report")
    follow_up_questions: list[str] = spec_field("follow_up_questions")
    sources: list[str] = spec_field("sources")


class WebSearchItemStructure(StructureBase):
    """A single web search to perform.

    Represents one web search query with rationale for its inclusion
    in a multi-query search plan.

    Attributes
    ----------
    reason : str
        Explanation for why this search is needed.
    query : str
        Web search query text.

    Examples
    --------
    >>> item = WebSearchItemStructure(
    ...     reason="Find latest news",
    ...     query="AI developments 2024"
    ... )
    """

    reason: str = spec_field("reason")
    query: str = spec_field("query")


class WebSearchItemResultStructure(StructureBase):
    """Result of a single web search.

    Contains the text content retrieved from executing one web search query.

    Attributes
    ----------
    text : str
        Retrieved text content from the web search.

    Examples
    --------
    >>> result = WebSearchItemResultStructure(text="Search result content")
    """

    text: str = spec_field("text")


class WebSearchPlanStructure(StructureBase):
    """Collection of web searches required to satisfy the query.

    Represents a plan containing multiple web searches that together
    provide comprehensive coverage for answering a user query.

    Attributes
    ----------
    searches : list[WebSearchItemStructure]
        List of web search queries to execute.

    Examples
    --------
    >>> plan = WebSearchPlanStructure(
    ...     searches=[WebSearchItemStructure(reason="R", query="Q")]
    ... )
    """

    searches: list[WebSearchItemStructure] = spec_field("searches")


class WebSearchStructure(StructureBase):
    """Complete output of a web search workflow.

    Represents the full lifecycle of a web search operation, from the
    original query through plan generation, execution, and final report.

    Attributes
    ----------
    query : str
        Original user query.
    web_search_plan : WebSearchPlanStructure
        Generated search plan.
    web_search_results : list[WebSearchItemResultStructure]
        List of search results.
    web_search_report : WebSearchReportStructure
        Final synthesized report.

    Methods
    -------
    print()
        Return the markdown report.

    Examples
    --------
    >>> workflow = WebSearchStructure(
    ...     query="Test query",
    ...     web_search_plan=WebSearchPlanStructure(searches=[]),
    ...     web_search_results=[],
    ...     web_search_report=WebSearchReportStructure(
    ...         short_summary="S",
    ...         markdown_report="R",
    ...         follow_up_questions=[],
    ...         sources=[]
    ...     )
    ... )
    """

    query: str = spec_field("query")
    web_search_plan: WebSearchPlanStructure = spec_field("web_search_plan")
    web_search_results: list[WebSearchItemResultStructure] = spec_field(
        "web_search_results"
    )
    web_search_report: WebSearchReportStructure = spec_field("web_search_report")

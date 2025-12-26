"""Shared structured output model for web search results."""

from __future__ import annotations

from typing import List

from .base import BaseStructure, spec_field


class WebSearchReportStructure(BaseStructure):
    """Structured output from the writer agent."""

    short_summary: str = spec_field("short_summary")
    markdown_report: str = spec_field("markdown_report")
    follow_up_questions: List[str] = spec_field("follow_up_questions")
    sources: List[str] = spec_field("sources")


class WebSearchItemStructure(BaseStructure):
    """A single web search to perform."""

    reason: str = spec_field("reason")
    query: str = spec_field("query")


class WebSearchItemResultStructure(BaseStructure):
    """Result of a single web search."""

    text: str = spec_field("text")


class WebSearchPlanStructure(BaseStructure):
    """Collection of searches required to satisfy the query."""

    searches: List[WebSearchItemStructure] = spec_field("searches")


class WebSearchStructure(BaseStructure):
    """Complete output of a web search workflow."""

    query: str = spec_field("query")
    web_search_plan: WebSearchPlanStructure = spec_field("web_search_plan")
    web_search_results: List[WebSearchItemResultStructure] = spec_field(
        "web_search_results"
    )
    web_search_report: WebSearchReportStructure = spec_field("web_search_report")

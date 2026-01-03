"""Generic base classes for search agent workflows.

This module provides abstract base classes that extract common patterns from
web search and vector search implementations, eliminating code duplication
and providing a consistent interface for new search types.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, List, Optional, TypeVar, Union

from ..base import AgentBase
from ..config import AgentConfig

# Type variables for search workflow components
ItemType = TypeVar("ItemType")  # Search item structure (e.g., WebSearchItemStructure)
ResultType = TypeVar("ResultType")  # Individual search result
PlanType = TypeVar("PlanType")  # Complete search plan structure
ReportType = TypeVar("ReportType")  # Final report structure
OutputType = TypeVar("OutputType")  # Generic output type


class SearchPlanner(AgentBase, Generic[PlanType]):
    """Generic planner agent for search workflows.

    Subclasses implement specific planner logic by overriding the
    `_configure_agent` method and specifying the output type.

    Methods
    -------
    run_agent(query)
        Generate a search plan for the provided query.
    _configure_agent()
        Return AgentConfig for this planner instance.
    """

    def __init__(
        self,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the planner agent.

        Parameters
        ----------
        prompt_dir : Path, optional
            Directory containing prompt templates.
        default_model : str, optional
            Default model identifier to use when not defined in config.
        """
        config = self._configure_agent()
        super().__init__(
            config=config,
            prompt_dir=prompt_dir,
            default_model=default_model,
        )

    @abstractmethod
    def _configure_agent(self) -> AgentConfig:
        """Return configuration for this planner.

        Returns
        -------
        AgentConfig
            Configuration with name, description, and output_type set.

        Examples
        --------
        >>> config = AgentConfig(
        ...     name="web_planner",
        ...     description="Plan web searches",
        ...     output_type=WebSearchPlanStructure,
        ... )
        >>> return config
        """
        pass

    async def run_agent(self, query: str) -> PlanType:
        """Generate a search plan for the query.

        Parameters
        ----------
        query : str
            User search query.

        Returns
        -------
        PlanType
            Generated search plan of the configured output type.
        """
        result: PlanType = await self.run_async(
            input=query,
            output_type=self._output_type,
        )
        return result


class SearchToolAgent(AgentBase, Generic[ItemType, ResultType, PlanType]):
    """Generic tool agent for executing search workflows.

    Executes individual searches in a plan with concurrency control.
    Subclasses implement search execution logic by overriding the
    `_configure_agent` and `run_search` methods.

    Methods
    -------
    run_agent(search_plan)
        Execute all searches in the plan.
    run_search(item)
        Execute a single search item.
    _configure_agent()
        Return AgentConfig for this tool agent.
    """

    def __init__(
        self,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
        max_concurrent_searches: int = 10,
    ) -> None:
        """Initialize the search tool agent.

        Parameters
        ----------
        prompt_dir : Path, optional
            Directory containing prompt templates.
        default_model : str, optional
            Default model identifier to use when not defined in config.
        max_concurrent_searches : int, default=10
            Maximum number of concurrent search operations.
        """
        self._max_concurrent_searches = max_concurrent_searches
        config = self._configure_agent()
        super().__init__(
            config=config,
            prompt_dir=prompt_dir,
            default_model=default_model,
        )

    @abstractmethod
    def _configure_agent(self) -> AgentConfig:
        """Return configuration for this tool agent.

        Returns
        -------
        AgentConfig
            Configuration with name, description, input_type, and tools set.

        Examples
        --------
        >>> config = AgentConfig(
        ...     name="web_search",
        ...     description="Perform web searches",
        ...     input_type=WebSearchPlanStructure,
        ...     tools=[WebSearchTool()],
        ... )
        >>> return config
        """
        pass

    @abstractmethod
    async def run_search(self, item: ItemType) -> ResultType:
        """Execute a single search item.

        Parameters
        ----------
        item : ItemType
            Individual search item from the plan.

        Returns
        -------
        ResultType
            Result of executing the search item.
        """
        pass

    async def run_agent(self, search_plan: PlanType) -> List[ResultType]:
        """Execute all searches in the plan with concurrency control.

        Parameters
        ----------
        search_plan : PlanType
            Plan structure containing search items.

        Returns
        -------
        list[ResultType]
            Completed search results from executing the plan.
        """
        semaphore = asyncio.Semaphore(self._max_concurrent_searches)

        async def _bounded_search(item: ItemType) -> Optional[ResultType]:
            """Execute search within concurrency limit."""
            async with semaphore:
                return await self.run_search(item)

        items = getattr(search_plan, "searches", [])
        tasks = [asyncio.create_task(_bounded_search(item)) for item in items]
        results = await asyncio.gather(*tasks)

        return [result for result in results if result is not None]


class SearchWriter(AgentBase, Generic[ReportType]):
    """Generic writer agent for search workflow reports.

    Synthesizes search results into a final report. Subclasses implement
    specific report generation logic by overriding the `_configure_agent` method.

    Methods
    -------
    run_agent(query, search_results)
        Generate a report from search results.
    _configure_agent()
        Return AgentConfig for this writer instance.
    """

    def __init__(
        self,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the writer agent.

        Parameters
        ----------
        prompt_dir : Path, optional
            Directory containing prompt templates.
        default_model : str, optional
            Default model identifier to use when not defined in config.
        """
        config = self._configure_agent()
        super().__init__(
            config=config,
            prompt_dir=prompt_dir,
            default_model=default_model,
        )

    @abstractmethod
    def _configure_agent(self) -> AgentConfig:
        """Return configuration for this writer.

        Returns
        -------
        AgentConfig
            Configuration with name, description, and output_type set.

        Examples
        --------
        >>> config = AgentConfig(
        ...     name="web_writer",
        ...     description="Write web search report",
        ...     output_type=WebSearchReportStructure,
        ... )
        >>> return config
        """
        pass

    async def run_agent(
        self,
        query: str,
        search_results: List[ResultType],
    ) -> ReportType:
        """Generate a report from search results.

        Parameters
        ----------
        query : str
            Original search query.
        search_results : list[ResultType]
            Results from the search execution phase.

        Returns
        -------
        ReportType
            Final report structure of the configured output type.
        """
        template_context = {
            "original_query": query,
            "search_results": search_results,
        }
        result: ReportType = await self.run_async(
            input=query,
            context=template_context,
            output_type=self._output_type,
        )
        return result


__all__ = [
    "SearchPlanner",
    "SearchToolAgent",
    "SearchWriter",
]

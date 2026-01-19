"""Generic base classes for search agent workflows.

This module provides abstract base classes that extract common patterns from
web search and vector search implementations, eliminating code duplication
and providing a consistent interface for new search types.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar, Union

from ..base import AgentBase
from ..configuration import AgentConfiguration
from ...structure.base import StructureBase

# Type variables for search workflow components
ItemType = TypeVar(
    "ItemType", bound=StructureBase
)  # Search item structure (e.g., WebSearchItemStructure)
ResultType = TypeVar("ResultType")  # Individual search result
PlanType = TypeVar("PlanType", bound=StructureBase)  # Complete search plan structure
ReportType = TypeVar("ReportType", bound=StructureBase)  # Final report structure


class SearchPlanner(AgentBase, Generic[PlanType]):
    """Generic planner agent for search workflows.

    Subclasses implement specific planner logic by overriding the
    `_configure_agent` method and specifying the output type.

    Parameters
    ----------
    template_path : Path | str | None, optional
        Template file path for prompt rendering.
    model : str | None, optional
        Model identifier to use when not defined in configuration.

    Methods
    -------
    run_agent(query)
        Generate a search plan for the provided query.
    _configure_agent()
        Return AgentConfiguration for this planner instance.

    Raises
    ------
    ValueError
        If the configuration omits a model identifier.

    Examples
    --------
    >>> class MyPlanner(SearchPlanner):
    ...     def _configure_agent(self, template_path=None, model=None):
    ...         return AgentConfiguration(
    ...             name="my_planner",
    ...             description="Plans searches",
    ...             output_structure=WebSearchPlanStructure,
    ...         )
    >>> planner = MyPlanner(model="gpt-4o-mini")
    """

    def __init__(
        self,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the planner agent."""
        configuration = self._configure_agent(
            template_path=template_path, model=model, **kwargs
        )
        super().__init__(configuration=configuration)

    @abstractmethod
    def _configure_agent(
        self,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentConfiguration:
        """Return configuration for this planner.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and output_structure set.

        Examples
        --------
        >>> configuration = AgentConfiguration(
        ...     name="web_planner",
        ...     description="Plan web searches",
        ...     output_structure=WebSearchPlanStructure,
        ... )
        >>> return configuration
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
            output_structure=self._output_structure,
        )
        return result


class SearchToolAgent(AgentBase, Generic[ItemType, ResultType, PlanType]):
    """Generic tool agent for executing search workflows.

    Executes individual searches in a plan with concurrency control.
    Subclasses implement search execution logic by overriding the
    `_configure_agent` and `run_search` methods.

    Parameters
    ----------
    template_path : Path | str | None, optional
        Template file path for prompt rendering.
    model : str | None, optional
        Model identifier to use when not defined in configuration.
    max_concurrent_searches : int, default=10
        Maximum number of concurrent search operations.

    Methods
    -------
    run_agent(search_plan)
        Execute all searches in the plan.
    run_search(item)
        Execute a single search item.
    _configure_agent()
        Return AgentConfiguration for this tool agent.

    Raises
    ------
    ValueError
        If the configuration omits a model identifier.

    Examples
    --------
    >>> class MyTool(SearchToolAgent):
    ...     def _configure_agent(self, *, template_path: Path | str | None = None, model: str | None = None):
    ...         return AgentConfiguration(
    ...             name="my_tool",
    ...             description="Executes searches",
    ...             input_structure=WebSearchPlanStructure,
    ...         )
    ...     async def run_search(self, item):
    ...         return "result"
    >>> tool = MyTool(model="gpt-4o-mini")
    """

    def __init__(
        self,
        *,
        template_path: Path | str | None = None,
        model: str | None = None,
        max_concurrent_searches: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the search tool agent."""
        self._max_concurrent_searches = max_concurrent_searches
        configuration = self._configure_agent(
            template_path=template_path,
            model=model,
            **kwargs,
        )
        super().__init__(configuration=configuration)

    @abstractmethod
    def _configure_agent(
        self,
        *,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentConfiguration:
        """Return configuration for this tool agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, input_structure, and tools set.

        Examples
        --------
        >>> configuration = AgentConfiguration(
        ...     name="web_search",
        ...     description="Perform web searches",
        ...     input_structure=WebSearchPlanStructure,
        ...     tools=[WebSearchTool()],
        ... )
        >>> return configuration
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

    Parameters
    ----------
    template_path : Path | str | None, optional
        Template file path for prompt rendering.
    model : str | None, optional
        Model identifier to use when not defined in configuration.

    Methods
    -------
    run_agent(query, search_results)
        Generate a report from search results.
    _configure_agent()
        Return AgentConfiguration for this writer instance.

    Raises
    ------
    ValueError
        If the configuration omits a model identifier.

    Examples
    --------
    >>> class MyWriter(SearchWriter):
    ...     def _configure_agent(self, template_path=None, model=None):
    ...         return AgentConfiguration(
    ...             name="my_writer",
    ...             description="Writes reports",
    ...             output_structure=WebSearchReportStructure,
    ...         )
    >>> writer = MyWriter(model="gpt-4o-mini")
    """

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
            output_structure=self._output_structure,
        )
        return result


__all__ = [
    "SearchPlanner",
    "SearchToolAgent",
    "SearchWriter",
]

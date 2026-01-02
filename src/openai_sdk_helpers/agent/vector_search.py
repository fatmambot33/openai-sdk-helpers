"""Core workflow management for ``vector search``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents import custom_span, gen_trace_id, trace

from ..structure.vector_search import (
    VectorSearchItemStructure,
    VectorSearchItemResultStructure,
    VectorSearchItemResultsStructure,
    VectorSearchStructure,
    VectorSearchPlanStructure,
    VectorSearchReportStructure,
)
from ..vector_storage import VectorStorage
from .base import AgentBase
from .config import AgentConfig
from .utils import run_coroutine_agent_sync

MAX_CONCURRENT_SEARCHES = 10


class VectorSearchPlanner(AgentBase):
    """Plan vector searches to satisfy a user query.

    Methods
    -------
    run_agent(query)
        Generate a vector search plan for the provided query.
    """

    def __init__(
        self, prompt_dir: Optional[Path] = None, default_model: Optional[str] = None
    ) -> None:
        """Initialize the planner agent.

        Parameters
        ----------
        prompt_dir : pathlib.Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.

        Returns
        -------
        None
        """
        config = AgentConfig(
            name="vector_planner",
            description="Plan vector searches based on a user query.",
            output_type=VectorSearchPlanStructure,
        )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )

    async def run_agent(self, query: str) -> VectorSearchPlanStructure:
        """Create a search plan for ``query``.

        Parameters
        ----------
        query : str
            User search query.

        Returns
        -------
        VectorSearchPlanStructure
            Generated search plan.
        """
        result: VectorSearchPlanStructure = await self.run_async(
            input=query,
            output_type=self._output_type,
        )

        return result


class VectorSearchTool(AgentBase):
    """Execute vector searches defined in a search plan.

    Methods
    -------
    run_agent(search_plan)
        Execute searches described by the plan.
    run_search(item)
        Perform a single vector search and summarise the result.
    """

    def __init__(
        self,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
        store_name: Optional[str] = None,
        max_concurrent_searches: int = MAX_CONCURRENT_SEARCHES,
        vector_storage: Optional[VectorStorage] = None,
        vector_storage_factory: Optional[Callable[[str], VectorStorage]] = None,
    ) -> None:
        """Initialize the search tool agent.

        Parameters
        ----------
        prompt_dir : pathlib.Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.
        store_name : str or None, default=None
            Name of the vector store to query.
        max_concurrent_searches : int, default=MAX_CONCURRENT_SEARCHES
            Maximum number of concurrent vector search tasks to run.
        vector_storage : VectorStorage or None, default=None
            Optional preconfigured vector storage instance to reuse.
        vector_storage_factory : callable, default=None
            Factory for constructing a :class:`VectorStorage` when one is not
            provided. Receives ``store_name`` as an argument.

        Returns
        -------
        None
        """
        self._vector_storage: Optional[VectorStorage] = None
        self._store_name = store_name or "editorial"
        self._vector_storage_factory = vector_storage_factory
        if vector_storage is not None:
            self._vector_storage = vector_storage
        self._max_concurrent_searches = max_concurrent_searches
        config = AgentConfig(
            name="vector_search",
            description="Perform vector searches based on a search plan.",
            input_type=VectorSearchPlanStructure,
            output_type=VectorSearchItemResultsStructure,
        )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )

    def _get_vector_storage(self) -> VectorStorage:
        """Return a cached vector storage instance.

        Returns
        -------
        VectorStorage
            Vector storage helper for executing searches.
        """
        if self._vector_storage is None:
            if self._vector_storage_factory is not None:
                self._vector_storage = self._vector_storage_factory(self._store_name)
            else:
                self._vector_storage = VectorStorage(store_name=self._store_name)
        return self._vector_storage

    async def run_agent(
        self, search_plan: VectorSearchPlanStructure
    ) -> VectorSearchItemResultsStructure:
        """Execute all searches in the plan with a progress bar.

        Parameters
        ----------
        search_plan : VectorSearchPlanStructure
            Plan describing each search to perform.

        Returns
        -------
        VectorSearchItemResultsStructure
            Collection of results for the completed searches.
        """
        with custom_span("Search vector store"):
            semaphore = asyncio.Semaphore(self._max_concurrent_searches)

            async def _bounded_search(
                item: VectorSearchItemStructure,
            ) -> VectorSearchItemResultStructure:
                """Execute a single search within the concurrency limit.

                Parameters
                ----------
                item : VectorSearchItemStructure
                    Search item to process.

                Returns
                -------
                VectorSearchItemResultStructure
                    Result of the search.
                """
                async with semaphore:
                    return await self.run_search(item)

            tasks = [
                asyncio.create_task(_bounded_search(item))
                for item in search_plan.searches
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            results = VectorSearchItemResultsStructure()
            for item, result in zip(search_plan.searches, results_list):
                if isinstance(result, BaseException):
                    results.errors.append(f"Search for '{item.query}' failed: {result}")
                    continue
                if result is not None:
                    results.append(result)

            return results

    async def run_search(
        self, item: VectorSearchItemStructure
    ) -> VectorSearchItemResultStructure:
        """Perform a single vector search using the search tool.

        Parameters
        ----------
        item : VectorSearchItemStructure
            Search item containing the query and reason.

        Returns
        -------
        VectorSearchItemResultStructure
            Summarized search result. The ``texts`` attribute is empty when no
            results are found.
        """
        results = self._get_vector_storage().search(item.query)
        if results is None:
            texts: List[str] = []
        else:
            texts = [
                content.text
                for result in results.data
                for content in (result.content or [])
                if getattr(content, "text", None)
            ]
        return VectorSearchItemResultStructure(texts=texts)


class VectorSearchWriter(AgentBase):
    """Generate reports summarizing vector search results.

    Methods
    -------
    run_agent(query, search_results)
        Compile a final report from search results.
    """

    def __init__(
        self, prompt_dir: Optional[Path] = None, default_model: Optional[str] = None
    ) -> None:
        """Initialize the writer agent.

        Parameters
        ----------
        prompt_dir : pathlib.Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.

        Returns
        -------
        None
        """
        config = AgentConfig(
            name="vector_writer",
            description="Write a report based on search results.",
            output_type=VectorSearchReportStructure,
        )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )

    async def run_agent(
        self, query: str, search_results: VectorSearchItemResultsStructure
    ) -> VectorSearchReportStructure:
        """Compile a final report from search results.

        Parameters
        ----------
        query : str
            Original search query.
        search_results : VectorSearchItemResultsStructure
            Results returned from the search step.

        Returns
        -------
        VectorSearchReportStructure
            Generated report for the query.
        """
        template_context: Dict[str, Any] = {
            "original_query": query,
            "search_results": search_results,
        }
        result: VectorSearchReportStructure = await self.run_async(
            input=query,
            context=template_context,
            output_type=self._output_type,
        )

        return result


class VectorSearch(AgentBase):
    """Manage the complete vector search workflow.

    This high-level agent orchestrates a multi-step research process that plans
    searches, executes them concurrently against a vector store, and generates
    comprehensive reports. It combines ``VectorSearchPlanner``,
    ``VectorSearchTool``, and ``VectorSearchWriter`` into a single workflow.

    Examples
    --------
    Basic vector search:

    >>> from pathlib import Path
    >>> from openai_sdk_helpers.agent.vector_search import VectorSearch
    >>> prompts = Path("./prompts")
    >>> search = VectorSearch(prompt_dir=prompts, default_model="gpt-4o-mini")
    >>> result = search.run_agent_sync("What are the key findings in recent AI research?")
    >>> print(result.report.report)

    Custom vector store:

    >>> from openai_sdk_helpers.vector_storage import VectorStorage
    >>> storage = VectorStorage(store_name="research_papers")
    >>> search = VectorSearch(
    ...     default_model="gpt-4o-mini",
    ...     vector_storage=storage,
    ...     max_concurrent_searches=5
    ... )

    Methods
    -------
    run_agent(search_query)
        Execute the research workflow asynchronously.
    run_agent_sync(search_query)
        Execute the research workflow synchronously.
    run_vector_agent(search_query)
        Convenience asynchronous entry point for the workflow.
    run_vector_agent_sync(search_query)
        Convenience synchronous entry point for the workflow.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
        vector_store_name: Optional[str] = None,
        max_concurrent_searches: int = MAX_CONCURRENT_SEARCHES,
        vector_storage: Optional[VectorStorage] = None,
        vector_storage_factory: Optional[Callable[[str], VectorStorage]] = None,
    ) -> None:
        """Create the main VectorSearch agent.

        Parameters
        ----------
        config : AgentConfig or None, default=None
            Optional configuration for the agent.
        prompt_dir : pathlib.Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.
        vector_store_name : str or None, default=None
            Name of the vector store to query.
        max_concurrent_searches : int, default=MAX_CONCURRENT_SEARCHES
            Maximum number of concurrent search tasks to run.
        vector_storage : VectorStorage or None, default=None
            Optional preconfigured vector storage instance to reuse.
        vector_storage_factory : callable, default=None
            Factory used to construct a :class:`VectorStorage` when one is not
            provided. Receives ``vector_store_name`` as an argument.

        Returns
        -------
        None
        """
        if config is None:
            config = AgentConfig(
                name="vector_agent",
                description="Coordinates the research process, including planning, searching, and report writing.",
                output_type=VectorSearchStructure,
                input_type=VectorSearchReportStructure,
            )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )
        self._prompt_dir = prompt_dir
        self._vector_store_name = vector_store_name
        self._max_concurrent_searches = max_concurrent_searches
        self._vector_storage = vector_storage
        self._vector_storage_factory = vector_storage_factory

    async def run_agent(self, search_query: str) -> VectorSearchStructure:
        """Execute the entire research workflow for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        VectorSearchStructure
            Completed research output.
        """
        trace_id = gen_trace_id()
        with trace("VectorSearch trace", trace_id=trace_id):
            planner = VectorSearchPlanner(
                prompt_dir=self._prompt_dir, default_model=self.model
            )
            tool = VectorSearchTool(
                prompt_dir=self._prompt_dir,
                default_model=self.model,
                store_name=self._vector_store_name,
                max_concurrent_searches=self._max_concurrent_searches,
                vector_storage=self._vector_storage,
                vector_storage_factory=self._vector_storage_factory,
            )
            writer = VectorSearchWriter(
                prompt_dir=self._prompt_dir, default_model=self.model
            )
            with custom_span("vector_search.plan"):
                search_plan = await planner.run_agent(query=search_query)
            with custom_span("vector_search.search"):
                search_results = await tool.run_agent(search_plan=search_plan)
            with custom_span("vector_search.write"):
                search_report = await writer.run_agent(search_query, search_results)
        return VectorSearchStructure(
            query=search_query,
            plan=search_plan,
            results=search_results,
            report=search_report,
        )

    def run_agent_sync(self, search_query: str) -> VectorSearchStructure:
        """Run :meth:`run_agent` synchronously for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        VectorSearchStructure
            Completed research output.
        """
        return run_coroutine_agent_sync(self.run_agent(search_query))

    @staticmethod
    async def run_vector_agent(search_query: str) -> VectorSearchStructure:
        """Return a research report for the given query using ``VectorSearch``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        VectorSearchStructure
            Completed research output.
        """
        return await VectorSearch().run_agent(search_query=search_query)

    @staticmethod
    def run_vector_agent_sync(search_query: str) -> VectorSearchStructure:
        """Run :meth:`run_vector_agent` synchronously for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        VectorSearchStructure
            Completed research output.
        """
        return run_coroutine_agent_sync(
            VectorSearch.run_vector_agent(search_query=search_query)
        )


__all__ = [
    "MAX_CONCURRENT_SEARCHES",
    "VectorSearchPlanner",
    "VectorSearchTool",
    "VectorSearchWriter",
    "VectorSearch",
]

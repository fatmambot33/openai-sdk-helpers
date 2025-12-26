"""Core workflow management for ``web search``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agents import custom_span, gen_trace_id, trace
from agents.model_settings import ModelSettings
from agents.tool import WebSearchTool

from ..structure.web_search import (
    WebSearchItemStructure,
    WebSearchItemResultStructure,
    WebSearchStructure,
    WebSearchPlanStructure,
    WebSearchReportStructure,
)
from .base import BaseAgent, _run_agent
from .config import AgentConfig
from .utils import run_coro_sync

MAX_CONCURRENT_SEARCHES = 10


class WebAgentPlanner(BaseAgent):
    """Plan web searches to satisfy a user query.

    Methods
    -------
    run_agent(query)
        Generate a search plan for the provided query.
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
            name="web_planner",
            description="Agent that plans web searches based on a user query.",
            output_type=WebSearchPlanStructure,
        )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )

    async def run_agent(self, query: str) -> WebSearchPlanStructure:
        """Plan searches for ``query``.

        Parameters
        ----------
        query : str
            User search query.

        Returns
        -------
        WebSearchPlanStructure
            Plan describing searches to perform.
        """
        result: WebSearchPlanStructure = await _run_agent(
            agent=self.get_agent(),
            agent_input=query,
            output_type=self._output_type,
        )

        return result


class WebSearchToolAgent(BaseAgent):
    """Execute web searches defined in a plan.

    Methods
    -------
    run_agent(search_plan)
        Execute searches described by the plan.
    run_search(item)
        Perform a single web search and summarise the result.
    """

    def __init__(
        self, prompt_dir: Optional[Path] = None, default_model: Optional[str] = None
    ) -> None:
        """Initialize the search tool agent.

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
            name="web_search",
            description="Agent that performs web searches and summarizes results.",
            input_type=WebSearchPlanStructure,
            tools=[WebSearchTool()],
            model_settings=ModelSettings(tool_choice="required"),
        )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )

    async def run_agent(
        self, search_plan: WebSearchPlanStructure
    ) -> List[WebSearchItemResultStructure]:
        """Execute all searches in the plan with a progress bar.

        Parameters
        ----------
        search_plan : WebSearchPlanStructure
            Plan describing each search to perform.

        Returns
        -------
        list[WebSearchItemResultStructure]
            Completed search results.
        """
        with custom_span("Search the web"):
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

            async def _bounded_search(
                item: WebSearchItemStructure,
            ) -> WebSearchItemResultStructure:
                """Execute a single search within the concurrency limit.

                Parameters
                ----------
                item : WebSearchItemStructure
                    Search item to process.

                Returns
                -------
                WebSearchItemResultStructure
                    Search result for ``item``.
                """
                async with semaphore:
                    return await self.run_search(item)

            tasks = [
                asyncio.create_task(_bounded_search(item))
                for item in search_plan.searches
            ]
            results = await asyncio.gather(*tasks)
            return [result for result in results if result is not None]

    async def run_search(
        self, item: WebSearchItemStructure
    ) -> WebSearchItemResultStructure:
        """Perform a single web search using the search agent.

        Parameters
        ----------
        item : WebSearchItemStructure
            Search item containing the query and reason.

        Returns
        -------
        WebSearchItemResultStructure
            Search result summarising the page.
        """
        template_context: Dict[str, Any] = {
            "search_term": item.query,
            "reason": item.reason,
        }

        result = await super().run(
            agent_input=item.query,
            agent_context=template_context,
            output_type=str,
        )
        return self._coerce_item_result(result)

    @staticmethod
    def _coerce_item_result(
        result: Union[str, WebSearchItemResultStructure, Any],
    ) -> WebSearchItemResultStructure:
        """Return a WebSearchItemResultStructure from varied agent outputs."""
        if isinstance(result, WebSearchItemResultStructure):
            return result
        try:
            return WebSearchItemResultStructure(text=str(result))
        except Exception:
            return WebSearchItemResultStructure(text="")


class WebAgentWriter(BaseAgent):
    """Summarize search results into a human-readable report.

    Methods
    -------
    run_agent(query, search_results)
        Compile a report from search results.
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
            name="web_writer",
            description="Agent that writes a report based on web search results.",
            output_type=WebSearchReportStructure,
        )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )

    async def run_agent(
        self, query: str, search_results: List[WebSearchItemResultStructure]
    ) -> WebSearchReportStructure:
        """Compile a report from search results.

        Parameters
        ----------
        query : str
            Original search query.
        search_results : list[WebSearchItemResultStructure]
            Results produced by the search step.

        Returns
        -------
        WebSearchReportStructure
            Generated report for the query.
        """
        template_context: Dict[str, Any] = {
            "original_query": query,
            "search_results": search_results,
        }
        result: WebSearchReportStructure = await _run_agent(
            agent=self.get_agent(),
            agent_input=query,
            agent_context=template_context,
            output_type=self._output_type,
        )

        return result


class WebAgentSearch(BaseAgent):
    """Manage the complete web search workflow.

    Methods
    -------
    run_agent(search_query)
        Execute the research workflow asynchronously.
    run_agent_sync(search_query)
        Execute the research workflow synchronously.
    run_web_agent(search_query)
        Convenience asynchronous entry point for the workflow.
    run_web_agent_sync(search_query)
        Convenience synchronous entry point for the workflow.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Create the main web search agent.

        Parameters
        ----------
        config : AgentConfig or None, default=None
            Optional configuration for the agent.
        prompt_dir : pathlib.Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.

        Returns
        -------
        None
        """
        if config is None:
            config = AgentConfig(
                name="web_agent",
                description="Agent that coordinates web searches and report writing.",
                output_type=WebSearchStructure,
            )
        super().__init__(
            config=config, prompt_dir=prompt_dir, default_model=default_model
        )
        self._prompt_dir = prompt_dir

    async def run_agent(self, search_query: str) -> WebSearchStructure:
        """Execute the entire research workflow for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        WebSearchStructure
            Completed research output.
        """
        trace_id = gen_trace_id()
        with trace("WebAgentSearch trace", trace_id=trace_id):
            planner = WebAgentPlanner(
                prompt_dir=self._prompt_dir, default_model=self.model
            )
            tool = WebSearchToolAgent(
                prompt_dir=self._prompt_dir, default_model=self.model
            )
            writer = WebAgentWriter(
                prompt_dir=self._prompt_dir, default_model=self.model
            )
            search_plan = await planner.run_agent(query=search_query)
            search_results = await tool.run_agent(search_plan=search_plan)
            search_report = await writer.run_agent(search_query, search_results)
        return WebSearchStructure(
            query=search_query,
            web_search_plan=search_plan,
            web_search_results=search_results,
            web_search_report=search_report,
        )

    def run_agent_sync(self, search_query: str) -> WebSearchStructure:
        """Run :meth:`run_agent` synchronously for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        WebSearchStructure
            Completed research output.
        """
        return run_coro_sync(self.run_agent(search_query))

    @staticmethod
    async def run_web_agent(search_query: str) -> WebSearchStructure:
        """Return a research report for the given query using ``WebAgentSearch``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        WebSearchStructure
            Completed research output.
        """
        return await WebAgentSearch().run_agent(search_query=search_query)

    @staticmethod
    def run_web_agent_sync(search_query: str) -> WebSearchStructure:
        """Run :meth:`run_web_agent` synchronously for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        WebSearchStructure
            Completed research output.
        """
        return run_coro_sync(WebAgentSearch.run_web_agent(search_query=search_query))


__all__ = [
    "MAX_CONCURRENT_SEARCHES",
    "WebAgentPlanner",
    "WebSearchToolAgent",
    "WebAgentWriter",
    "WebAgentSearch",
]

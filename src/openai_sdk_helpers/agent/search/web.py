"""Core workflow management for ``web search``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agents import custom_span, gen_trace_id, trace
from agents.model_settings import ModelSettings
from agents.tool import WebSearchTool

from ...structure.web_search import (
    WebSearchItemStructure,
    WebSearchItemResultStructure,
    WebSearchStructure,
    WebSearchPlanStructure,
    WebSearchReportStructure,
)
from ..config import AgentConfig
from ..utils import run_coroutine_agent_sync
from .base import SearchPlanner, SearchToolAgent, SearchWriter

MAX_CONCURRENT_SEARCHES = 10


class WebAgentPlanner(SearchPlanner[WebSearchPlanStructure]):
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
        prompt_dir : Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.
        """
        super().__init__(prompt_dir=prompt_dir, default_model=default_model)

    def _configure_agent(self) -> AgentConfig:
        """Return configuration for the web planner agent.

        Returns
        -------
        AgentConfig
            Configuration with name, description, and output type.
        """
        return AgentConfig(
            name="web_planner",
            description="Agent that plans web searches based on a user query.",
            output_type=WebSearchPlanStructure,
        )


class WebSearchToolAgent(
    SearchToolAgent[
        WebSearchItemStructure, WebSearchItemResultStructure, WebSearchPlanStructure
    ]
):
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
        prompt_dir : Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.
        """
        super().__init__(
            prompt_dir=prompt_dir,
            default_model=default_model,
            max_concurrent_searches=MAX_CONCURRENT_SEARCHES,
        )

    def _configure_agent(self) -> AgentConfig:
        """Return configuration for the web search tool agent.

        Returns
        -------
        AgentConfig
            Configuration with name, description, input type, and tools.
        """
        return AgentConfig(
            name="web_search",
            description="Agent that performs web searches and summarizes results.",
            input_type=WebSearchPlanStructure,
            tools=[WebSearchTool()],
            model_settings=ModelSettings(tool_choice="required"),
        )

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
            Search result summarizing the page.
        """
        with custom_span("Search the web"):
            template_context: Dict[str, Any] = {
                "search_term": item.query,
                "reason": item.reason,
            }

            result = await super(SearchToolAgent, self).run_async(
                input=item.query,
                context=template_context,
                output_type=str,
            )
            return self._coerce_item_result(result)

    @staticmethod
    def _coerce_item_result(
        result: Union[str, WebSearchItemResultStructure, Any],
    ) -> WebSearchItemResultStructure:
        """Return a WebSearchItemResultStructure from varied agent outputs.

        Parameters
        ----------
        result : str or WebSearchItemResultStructure or Any
            Agent output that may be of various types.

        Returns
        -------
        WebSearchItemResultStructure
            Coerced search result structure.
        """
        if isinstance(result, WebSearchItemResultStructure):
            return result
        try:
            return WebSearchItemResultStructure(text=str(result))
        except Exception:
            return WebSearchItemResultStructure(text="")


class WebAgentWriter(SearchWriter[WebSearchReportStructure]):
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
        prompt_dir : Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.
        """
        super().__init__(prompt_dir=prompt_dir, default_model=default_model)

    def _configure_agent(self) -> AgentConfig:
        """Return configuration for the web writer agent.

        Returns
        -------
        AgentConfig
            Configuration with name, description, and output type.
        """
        return AgentConfig(
            name="web_writer",
            description="Agent that writes a report based on web search results.",
            output_type=WebSearchReportStructure,
        )


class WebAgentSearch:
    """Manage the complete web search workflow.

    Methods
    -------
    run_agent_async(search_query)
        Execute the research workflow asynchronously.
    run_agent_sync(search_query)
        Execute the research workflow synchronously.
    run_web_agent_async(search_query)
        Convenience asynchronous entry point for the workflow.
    run_web_agent_sync(search_query)
        Convenience synchronous entry point for the workflow.
    """

    def __init__(
        self,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Create the main web search agent.

        Parameters
        ----------
        prompt_dir : Path or None, default=None
            Directory containing prompt templates.
        default_model : str or None, default=None
            Default model identifier to use when not defined in config.
        """
        self._prompt_dir = prompt_dir
        self._default_model = default_model

    async def run_agent_async(self, search_query: str) -> WebSearchStructure:
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
                prompt_dir=self._prompt_dir, default_model=self._default_model
            )
            tool = WebSearchToolAgent(
                prompt_dir=self._prompt_dir, default_model=self._default_model
            )
            writer = WebAgentWriter(
                prompt_dir=self._prompt_dir, default_model=self._default_model
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
        """Run :meth:`run_agent_async` synchronously for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        WebSearchStructure
            Completed research output.
        """
        return run_coroutine_agent_sync(self.run_agent_async(search_query))

    async def run_web_agent_async(self, search_query: str) -> WebSearchStructure:
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
        return await self.run_agent_async(search_query=search_query)

    @staticmethod
    def run_web_agent_sync(search_query: str) -> WebSearchStructure:
        """Run :meth:`run_web_agent_async` synchronously for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        WebSearchStructure
            Completed research output.
        """
        return run_coroutine_agent_sync(
            WebAgentSearch().run_web_agent_async(search_query=search_query)
        )


__all__ = [
    "MAX_CONCURRENT_SEARCHES",
    "WebAgentPlanner",
    "WebSearchToolAgent",
    "WebAgentWriter",
    "WebAgentSearch",
]

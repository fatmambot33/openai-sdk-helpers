"""Core workflow management for ``web search``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from agents import custom_span, gen_trace_id, trace
from agents.model_settings import ModelSettings
from agents.tool import WebSearchTool

from ...structure.prompt import PromptStructure
from ..base import AgentBase
from ...structure.web_search import (
    WebSearchItemStructure,
    WebSearchItemResultStructure,
    WebSearchStructure,
    WebSearchPlanStructure,
    WebSearchReportStructure,
)
from ..configuration import AgentConfiguration
from ..utils import run_coroutine_agent_sync
from .base import SearchPlanner, SearchToolAgent, SearchWriter
from ...tools import ToolSpec, ToolHandlerRegistration

MAX_CONCURRENT_SEARCHES = 10


class WebAgentPlanner(SearchPlanner[WebSearchPlanStructure]):
    """Plan web searches to satisfy a user query.

    Parameters
    ----------
    template_path : Path | str | None, default=None
        Template file path for prompt rendering.
    model : str | None, default=None
        Model identifier to use when not defined in configuration.

    Methods
    -------
    run_agent(query)
        Generate a search plan for the provided query.

    Raises
    ------
    ValueError
        If the configuration omits a model identifier.

    Examples
    --------
    >>> planner = WebAgentPlanner(model="gpt-4o-mini")
    """

    def _configure_agent(
        self,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentConfiguration:
        """Return configuration for the web planner agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and output type.
        """
        return AgentConfiguration(
            name="web_planner",
            instructions="Agent instructions",
            description="Agent that plans web searches based on a user query.",
            template_path=template_path,
            model=model,
            output_structure=WebSearchPlanStructure,
        )


class WebSearchToolAgent(
    SearchToolAgent[
        WebSearchItemStructure, WebSearchItemResultStructure, WebSearchPlanStructure
    ]
):
    """Execute web searches defined in a plan.

    Parameters
    ----------
    template_path : Path | str | None, default=None
        Template file path for prompt rendering.
    model : str or None, default=None
        Model identifier to use when not defined in configuration.

    Methods
    -------
    run_agent(search_plan)
        Execute searches described by the plan.
    run_search(item)
        Perform a single web search and summarise the result.

    Raises
    ------
    ValueError
        If the model is not provided.

    Examples
    --------
    >>> tool = WebSearchToolAgent(model="gpt-4o-mini")
    """

    def _configure_agent(
        self,
        *,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentConfiguration:
        """Return configuration for the web search tool agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, input type, and tools.
        """
        return AgentConfiguration(
            name="web_search",
            instructions="Agent instructions",
            description="Agent that performs web searches and summarizes results.",
            template_path=template_path,
            model=model,
            input_structure=WebSearchPlanStructure,
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

    Parameters
    ----------
    template_path : Path | str | None, default=None
        Template file path for prompt rendering.
    model : str | None, default=None
        Model identifier to use when not defined in configuration.

    Methods
    -------
    run_agent(query, search_results)
        Compile a report from search results.

    Raises
    ------
    ValueError
        If the configuration omits a model identifier.

    Examples
    --------
    >>> writer = WebAgentWriter(model="gpt-4o-mini")
    """

    def __init__(
        self,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the writer agent."""
        configuration = self._configure_agent(
            template_path=template_path, model=model, **kwargs
        )
        super().__init__(configuration=configuration)

    def _configure_agent(
        self,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentConfiguration:
        """Return configuration for the web writer agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and output type.
        """
        return AgentConfiguration(
            name="web_writer",
            instructions="Agent instructions",
            description="Agent that writes a report based on web search results.",
            template_path=template_path,
            model=model,
            output_structure=WebSearchReportStructure,
        )


class WebAgentSearch(AgentBase):
    """Manage the complete web search workflow.

    Parameters
    ----------
    template_path : Path | str | None, default=None
        Template file path for prompt rendering.
    model : str | None, default=None
        Model identifier to use when not defined in configuration.

    Methods
    -------
    run_agent_async(search_query)
        Execute the research workflow asynchronously.
    run_agent_sync(search_query)
        Execute the research workflow synchronously.
    as_response_tool(tool_name, tool_description)
        Build a Responses API tool definition and handler.
    run_web_agent_async(search_query)
        Convenience asynchronous entry point for the workflow.
    run_web_agent_sync(search_query)
        Convenience synchronous entry point for the workflow.

    Raises
    ------
    ValueError
        If the model identifier is not provided.

    Examples
    --------
    >>> search = WebAgentSearch(model="gpt-4o-mini")
    """

    def __init__(
        self,
        *,
        template_path: Path | str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the web search orchestration agent.

        Parameters
        ----------
        template_path : Path | str | None, default=None
            Optional template file path for prompt rendering.
        model : str | None, default=None
            Model identifier to use when not defined in configuration.
        """
        configuration = AgentConfiguration(
            name="web_agent_search",
            instructions="Agent instructions",
            description="Run a multi-step web search workflow.",
            template_path=template_path,
            model=model,
        )
        super().__init__(configuration=configuration)

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
                template_path=self._template_path, model=self.model
            )
            tool = WebSearchToolAgent(
                template_path=self._template_path, model=self.model
            )
            writer = WebAgentWriter(template_path=self._template_path, model=self.model)
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
        """Execute the entire research workflow for ``search_query`` synchronously.

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

    def as_tool_registration(
        self, tool_name: str, tool_description: str
    ) -> ToolHandlerRegistration:
        """Build a Responses API tool definition and handler for the web search agent.

        Parameters
        ----------
        tool_name : str
            Name of the tool.
        tool_description : str
            Description of the tool.

        Returns
        -------
        ToolHandlerRegistration
            Tool definition and handler for the Responses API.
        """
        tool_spec = ToolSpec(
            input_structure=PromptStructure,
            tool_name=tool_name,
            tool_description=tool_description,
            output_structure=WebSearchStructure,
        )
        return ToolHandlerRegistration(
            handler=self.run_agent_sync,
            tool_spec=tool_spec,
        )


__all__ = [
    "MAX_CONCURRENT_SEARCHES",
    "WebAgentPlanner",
    "WebSearchToolAgent",
    "WebAgentWriter",
    "WebAgentSearch",
]

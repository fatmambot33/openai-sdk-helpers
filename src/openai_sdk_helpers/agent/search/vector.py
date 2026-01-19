"""Core workflow management for ``vector search``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

from agents import custom_span, gen_trace_id, trace
from agents.model_settings import ModelSettings

from ...environment import DEFAULT_PROMPT_DIR
from ...structure.vector_search import (
    VectorSearchItemStructure,
    VectorSearchItemResultStructure,
    VectorSearchItemResultsStructure,
    VectorSearchStructure,
    VectorSearchPlanStructure,
    VectorSearchReportStructure,
)
from ...vector_storage import VectorStorage
from ..configuration import AgentConfiguration
from ..utils import run_coroutine_agent_sync
from .base import SearchPlanner, SearchToolAgent, SearchWriter
from ..base import AgentBase

MAX_CONCURRENT_SEARCHES = 10


class VectorAgentPlanner(SearchPlanner[VectorSearchPlanStructure]):
    """Plan vector searches to satisfy a user query.

    Parameters
    ----------
    template_path : Path | str | None, optional
        Template file path for prompt rendering.
    model : str | None, optional
        Model identifier to use when not defined in configuration.

    Methods
    -------
    run_agent(query)
        Generate a vector search plan for the provided query.

    Raises
    ------
    ValueError
        If the configuration omits a model identifier.

    Examples
    --------
    >>> planner = VectorSearchPlanner(model="gpt-4o-mini")
    """

    def _configure_agent(
        self,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentConfiguration:
        """Return configuration for the vector planner agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and output type.
        """
        return AgentConfiguration(
            name="vector_planner",
            instructions="Agent instructions",
            description="Plan vector searches based on a user query.",
            output_structure=VectorSearchPlanStructure,
            model_settings=ModelSettings(tool_choice="none"),
            template_path=template_path,
            model=model,
        )


class VectorSearchTool(
    SearchToolAgent[
        VectorSearchItemStructure,
        VectorSearchItemResultStructure,
        VectorSearchPlanStructure,
    ]
):
    """Execute vector searches defined in a search plan.

    Parameters
    ----------
    template_path : Path | str | None, optional
        Template file path for prompt rendering.
    model : str | None, optional
        Model identifier to use when not defined in configuration.
    store_name : str
        Name of the vector store to query.
    max_concurrent_searches : int, default=MAX_CONCURRENT_SEARCHES
        Maximum number of concurrent vector search tasks to run.
    vector_storage : VectorStorage or None, default=None
        Optional preconfigured vector storage instance to reuse.
    vector_storage_factory : Callable or None, default=None
        Factory for constructing a VectorStorage when one is not provided.
        Receives ``store_name`` as an argument.

    Methods
    -------
    run_agent(search_plan)
        Execute searches described by the plan.
    run_search(item)
        Perform a single vector search and summarise the result.

    Raises
    ------
    ValueError
        If the configuration omits a model identifier.

    Examples
    --------
    >>> tool = VectorSearchTool(model="gpt-4o-mini", store_name="my_store")
    """

    def _configure_agent(
        self,
        *,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentConfiguration:
        """Return configuration for the vector search tool agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and input type.
        """
        if self._store_name is None:
            raise ValueError("store_name must be provided to configure the agent.")
        return AgentConfiguration(
            name="vector_search",
            instructions="Agent instructions",
            description="Perform vector searches based on a search plan.",
            input_structure=VectorSearchPlanStructure,
            output_structure=VectorSearchItemResultsStructure,
            model_settings=ModelSettings(tool_choice="none"),
            template_path=template_path,
            model=model,
        )

    def __init__(
        self,
        *,
        store_name: str,
        template_path: Path | str | None = None,
        model: str | None = None,
        max_concurrent_searches: int = MAX_CONCURRENT_SEARCHES,
        vector_storage: Optional[VectorStorage] = None,
        vector_storage_factory: Optional[Callable[[str], VectorStorage]] = None,
    ) -> None:
        """Initialize the vector search tool agent."""
        self._vector_storage = vector_storage
        self._vector_storage_factory = vector_storage_factory
        self._store_name = store_name
        super().__init__(
            template_path=template_path,
            model=model,
            max_concurrent_searches=max_concurrent_searches,
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


class VectorSearchWriter(SearchWriter[VectorSearchReportStructure]):
    """Generate reports summarizing vector search results.

    Parameters
    ----------
    template_path : Path | str | None, optional
        Template file path for prompt rendering.
    model : str | None, optional
        Model identifier to use when not defined in configuration.

    Methods
    -------
    run_agent(query, search_results)
        Compile a final report from search results.

    Raises
    ------
    ValueError
        If the configuration omits a model identifier.

    Examples
    --------
    >>> writer = VectorSearchWriter(model="gpt-4o-mini")
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
        *,
        template_path: Path | str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentConfiguration:
        """Return configuration for the vector writer agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and output type.
        """
        return AgentConfiguration(
            name="vector_writer",
            instructions="Agent instructions",
            description="Write a report based on search results.",
            output_structure=VectorSearchReportStructure,
            model_settings=ModelSettings(tool_choice="none"),
            template_path=template_path,
            model=model,
        )


class VectorAgentSearch(AgentBase):
    """Manage the complete vector search workflow.

    This high-level agent orchestrates a multi-step research process that plans
    searches, executes them concurrently against a vector store, and generates
    comprehensive reports. It combines ``VectorSearchPlanner``,
    ``VectorSearchTool``, and ``VectorSearchWriter`` into a single workflow.

    Parameters
    ----------
    prompt_dir : Path or None, default=None
        Directory containing prompt templates. Defaults to the packaged
        ``prompt`` directory when not provided.
    default_model : str or None, default=None
        Default model identifier to use when not defined in configuration.
    vector_store_name : str or None, default=None
        Name of the vector store to query.
    max_concurrent_searches : int, default=MAX_CONCURRENT_SEARCHES
        Maximum number of concurrent search tasks to run.
    vector_storage : VectorStorage or None, default=None
        Optional preconfigured vector storage instance to reuse.
    vector_storage_factory : callable, default=None
        Factory used to construct a VectorStorage when one is not provided.
        Receives ``vector_store_name`` as an argument.

    Examples
    --------
    Basic vector search:

    >>> from pathlib import Path
    >>> from openai_sdk_helpers.agent.search.vector import VectorSearch
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
    as_response_tool(vector_store_name, tool_name, tool_description)
        Build a Responses API tool definition and handler.
    run_vector_agent(search_query)
        Convenience asynchronous entry point for the workflow.
    run_vector_agent_sync(search_query)
        Convenience synchronous entry point for the workflow.

    Raises
    ------
    ValueError
        If the model identifier is not provided.
    """

    def __init__(
        self,
        *,
        vector_store_name: str,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
        max_concurrent_searches: int = MAX_CONCURRENT_SEARCHES,
        vector_storage: Optional[VectorStorage] = None,
        vector_storage_factory: Optional[Callable[[str], VectorStorage]] = None,
    ) -> None:
        """Create the main VectorSearch agent."""
        self._prompt_dir = prompt_dir or DEFAULT_PROMPT_DIR
        self._default_model = default_model
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
            planner = VectorAgentPlanner(
                template_path=self._prompt_dir, model=self._default_model
            )
            tool = VectorSearchTool(
                template_path=self._prompt_dir,
                model=self._default_model,
                max_concurrent_searches=self._max_concurrent_searches,
                store_name=self._vector_store_name,
            )
            writer = VectorSearchWriter(
                template_path=self._prompt_dir, model=self._default_model
            )
            with custom_span("vector_search.plan"):
                search_plan = await planner.run_agent(query=search_query)
            with custom_span("vector_search.search"):
                search_results_list = await tool.run_agent(search_plan=search_plan)
            with custom_span("vector_search.write"):
                search_report = await writer.run_agent(
                    search_query, search_results_list
                )
        search_results = VectorSearchItemResultsStructure(
            item_results=search_results_list
        )
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


__all__ = [
    "MAX_CONCURRENT_SEARCHES",
    "VectorAgentPlanner",
    "VectorSearchTool",
    "VectorSearchWriter",
    "VectorAgentSearch",
]

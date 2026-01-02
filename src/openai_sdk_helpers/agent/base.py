"""Base agent helpers built on the OpenAI Agents SDK."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from agents import Agent, Runner, RunResult, RunResultStreaming
from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from jinja2 import Template
from .runner import run_sync, run_streamed, run_async


class AgentConfigLike(Protocol):
    """Protocol describing the configuration attributes for AgentBase."""

    name: str
    description: Optional[str]
    model: Optional[str]
    template_path: Optional[str]
    input_type: Optional[Any]
    output_type: Optional[Any]
    tools: Optional[Any]
    model_settings: Optional[Any]


class AgentBase:
    """Factory for creating and configuring specialized agents.

    ``AgentBase`` provides the foundation for building OpenAI agents with support
    for Jinja2 prompt templates, custom tools, and both synchronous and
    asynchronous execution modes. All specialized agents in this package extend
    this base class.

    Examples
    --------
    Create a basic agent from configuration:

    >>> from openai_sdk_helpers.agent import AgentBase, AgentConfig
    >>> config = AgentConfig(
    ...     name="my_agent",
    ...     description="A custom agent",
    ...     model="gpt-4o-mini"
    ... )
    >>> agent = AgentBase(config=config, default_model="gpt-4o-mini")
    >>> result = agent.run_sync("What is 2+2?")

    Use async execution:

    >>> import asyncio
    >>> async def main():
    ...     result = await agent.run_async("Explain quantum physics")
    ...     return result
    >>> asyncio.run(main())

    Methods
    -------
    from_config(config, run_context_wrapper)
        Instantiate a ``AgentBase`` from configuration.
    build_prompt_from_jinja(run_context_wrapper)
        Render the agent prompt using Jinja and optional context.
    get_prompt(run_context_wrapper, _)
        Render the agent prompt using the provided run context.
    get_agent()
        Construct the configured :class:`agents.Agent` instance.
    run(input, context, output_type)
        Execute the agent asynchronously (alias of ``run_async``).
    run_async(input, context, output_type)
        Execute the agent asynchronously and optionally cast the result.
    run_sync(input, context, output_type)
        Execute the agent synchronously.
    run_streamed(input, context, output_type)
        Return a streaming result for the agent execution.
    as_tool()
        Return the agent as a callable tool.
    """

    def __init__(
        self,
        config: AgentConfigLike,
        run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the ``AgentBase`` using a configuration object.

        Parameters
        ----------
        config
            Configuration describing this agent.
        run_context_wrapper
            Optional wrapper providing runtime context for prompt rendering.
            Default ``None``.
        prompt_dir
            Optional directory holding prompt templates.
        default_model
            Optional fallback model identifier if the config does not supply one.

        Returns
        -------
        None
        """
        name = config.name
        description = config.description or ""
        model = config.model or default_model
        if not model:
            raise ValueError("Model is required to construct the agent.")

        prompt_path: Optional[Path]
        if config.template_path:
            prompt_path = Path(config.template_path)
        elif prompt_dir is not None:
            prompt_path = prompt_dir / f"{name}.jinja"
        else:
            prompt_path = None

        if prompt_path is None:
            self._template = Template("")
        elif prompt_path.exists():
            self._template = Template(prompt_path.read_text())
        else:
            raise FileNotFoundError(
                f"Prompt template for agent '{name}' not found at {prompt_path}."
            )

        self.agent_name = name
        self.description = description
        self.model = model

        self._input_type = config.input_type
        self._output_type = config.output_type or config.input_type
        self._tools = config.tools
        self._model_settings = config.model_settings
        self._run_context_wrapper = run_context_wrapper

    @classmethod
    def from_config(
        cls,
        config: AgentConfigLike,
        run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> "AgentBase":
        """Create a :class:`AgentBase` instance from configuration.

        Parameters
        ----------
        config
            Configuration describing the agent.
        run_context_wrapper
            Optional wrapper providing runtime context. Default ``None``.
        prompt_dir
            Optional directory holding prompt templates.
        default_model
            Optional fallback model identifier.

        Returns
        -------
        AgentBase
            Instantiated agent.
        """
        return cls(
            config=config,
            run_context_wrapper=run_context_wrapper,
            prompt_dir=prompt_dir,
            default_model=default_model,
        )

    def _build_prompt_from_jinja(self) -> str:
        """Return the rendered instructions prompt for this agent.

        Returns
        -------
        str
            Prompt text rendered from the Jinja template.
        """
        return self.build_prompt_from_jinja(
            run_context_wrapper=self._run_context_wrapper
        )

    def build_prompt_from_jinja(
        self, run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None
    ) -> str:
        """Render the agent prompt using the provided run context.

        Parameters
        ----------
        run_context_wrapper
            Wrapper whose ``context`` dictionary is used to render the Jinja
            template. Default ``None``.

        Returns
        -------
        str
            Rendered prompt text.
        """
        context = {}
        if run_context_wrapper is not None:
            context = run_context_wrapper.context

        return self._template.render(context)

    def get_prompt(
        self, run_context_wrapper: RunContextWrapper[Dict[str, Any]], _: Agent
    ) -> str:
        """Render the agent prompt using the provided run context.

        Parameters
        ----------
        run_context_wrapper
            Wrapper around the current run context whose ``context`` dictionary
            is used to render the Jinja template.
        _
            Underlying :class:`agents.Agent` instance (ignored).

        Returns
        -------
        str
            The rendered prompt.
        """
        return self.build_prompt_from_jinja(run_context_wrapper)

    def get_agent(self) -> Agent:
        """Construct and return the configured :class:`agents.Agent` instance.

        Returns
        -------
        Agent
            Initialized agent ready for execution.
        """
        agent_config: Dict[str, Any] = {
            "name": self.agent_name,
            "instructions": self._build_prompt_from_jinja() or ".",
            "model": self.model,
        }
        if self._output_type:
            agent_config["output_type"] = self._output_type
        if self._tools:
            agent_config["tools"] = self._tools
        if self._model_settings:
            agent_config["model_settings"] = self._model_settings

        return Agent(**agent_config)

    async def run_async(
        self,
        input: str,
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
    ) -> Any:
        """Execute the agent asynchronously.

        Parameters
        ----------
        input
            Prompt or query for the agent.
        context
            Optional dictionary passed to the agent. Default ``None``.
        output_type
            Optional type used to cast the final output. Default ``None``.

        Returns
        -------
        Any
            Agent result, optionally converted to ``output_type``.
        """
        if self._output_type is not None and output_type is None:
            output_type = self._output_type
        return await run_async(
            agent=self.get_agent(),
            input=input,
            context=context,
            output_type=output_type,
        )

    def run_sync(
        self,
        input: str,
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
    ) -> Any:
        """Run the agent synchronously.

        Parameters
        ----------
        input
            Prompt or query for the agent.
        context
            Optional dictionary passed to the agent. Default ``None``.
        output_type
            Optional type used to cast the final output. Default ``None``.

        Returns
        -------
        Any
            Agent result, optionally converted to ``output_type``.
        """
        return run_sync(
            agent=self.get_agent(),
            input=input,
            context=context,
            output_type=output_type,
        )

    def run_streamed(
        self,
        input: str,
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
    ) -> RunResultStreaming:
        """Return a streaming result for the agent execution.

        Parameters
        ----------
        input
            Prompt or query for the agent.
        context
            Optional dictionary passed to the agent. Default ``None``.
        output_type
            Optional type used to cast the final output. Default ``None``.

        Returns
        -------
        RunResultStreaming
            Streaming output wrapper from the agent execution.
        """
        result = run_streamed(
            agent=self.get_agent(),
            input=input,
            context=context,
        )
        if self._output_type and not output_type:
            output_type = self._output_type
        if output_type:
            return result.final_output_as(output_type)
        return result

    def as_tool(self) -> FunctionTool:
        """Return the agent as a callable tool.

        Returns
        -------
        FunctionTool
            Tool instance wrapping this agent.
        """
        agent = self.get_agent()
        tool_obj: FunctionTool = agent.as_tool(
            tool_name=self.agent_name, tool_description=self.description
        )  # type: ignore
        return tool_obj


__all__ = ["AgentConfigLike", "AgentBase"]

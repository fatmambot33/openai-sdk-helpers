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


class AgentConfigLike(Protocol):
    """Protocol describing the configuration attributes for BaseAgent."""

    name: str
    description: Optional[str]
    model: Optional[str]
    template_path: Optional[str]
    input_type: Optional[Any]
    output_type: Optional[Any]
    tools: Optional[Any]
    model_settings: Optional[Any]


class BaseAgent:
    """Factory for creating and configuring specialized agents.

    Methods
    -------
    from_config(config, run_context_wrapper)
        Instantiate a ``BaseAgent`` from configuration.
    build_prompt_from_jinja(run_context_wrapper)
        Render the agent prompt using Jinja and optional context.
    get_prompt(run_context_wrapper, _)
        Render the agent prompt using the provided run context.
    get_agent()
        Construct the configured :class:`agents.Agent` instance.
    run(agent_input, agent_context, output_type)
        Execute the agent asynchronously and optionally cast the result.
    run_sync(agent_input, agent_context, output_type)
        Execute the agent synchronously.
    run_streamed(agent_input, agent_context, output_type)
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
        """Initialize the ``BaseAgent`` using a configuration object.

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

        if prompt_path is not None and prompt_path.exists():
            self._template = Template(prompt_path.read_text())
        else:
            self._template = Template("")

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
    ) -> "BaseAgent":
        """Create a :class:`BaseAgent` instance from configuration.

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
        BaseAgent
            Instantiated agent.
        """
        return cls(
            config=config,
            run_context_wrapper=run_context_wrapper,
            prompt_dir=prompt_dir,
            default_model=default_model,
        )

    def _build_instructions_from_jinja(self) -> str:
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
            "instructions": self._build_instructions_from_jinja(),
            "model": self.model,
        }
        if self._output_type:
            agent_config["output_type"] = self._output_type
        if self._tools:
            agent_config["tools"] = self._tools
        if self._model_settings:
            agent_config["model_settings"] = self._model_settings

        return Agent(**agent_config)

    async def run(
        self,
        agent_input: str,
        agent_context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
    ) -> Any:
        """Execute the agent asynchronously.

        Parameters
        ----------
        agent_input
            Prompt or query for the agent.
        agent_context
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
        return await _run_agent(
            agent=self.get_agent(),
            agent_input=agent_input,
            agent_context=agent_context,
            output_type=output_type,
        )

    def run_sync(
        self,
        agent_input: str,
        agent_context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
    ) -> Any:
        """Run the agent synchronously.

        Parameters
        ----------
        agent_input
            Prompt or query for the agent.
        agent_context
            Optional dictionary passed to the agent. Default ``None``.
        output_type
            Optional type used to cast the final output. Default ``None``.

        Returns
        -------
        Any
            Agent result, optionally converted to ``output_type``.
        """
        result = _run_agent_sync(
            self.get_agent(),
            agent_input,
            agent_context=agent_context,
        )
        if self._output_type and not output_type:
            output_type = self._output_type
        if output_type:
            return result.final_output_as(output_type)
        return result

    def run_streamed(
        self,
        agent_input: str,
        agent_context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
    ) -> RunResultStreaming:
        """Return a streaming result for the agent execution.

        Parameters
        ----------
        agent_input
            Prompt or query for the agent.
        agent_context
            Optional dictionary passed to the agent. Default ``None``.
        output_type
            Optional type used to cast the final output. Default ``None``.

        Returns
        -------
        RunResultStreaming
            Streaming output wrapper from the agent execution.
        """
        result = _run_agent_streamed(
            agent=self.get_agent(),
            agent_input=agent_input,
            context=agent_context,
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


async def _run_agent(
    agent: Agent,
    agent_input: str,
    agent_context: Optional[Dict[str, Any]] = None,
    output_type: Optional[Any] = None,
) -> Any:
    """Run an ``Agent`` asynchronously.

    Parameters
    ----------
    agent
        Configured agent instance to execute.
    agent_input
        Prompt or query string for the agent.
    agent_context
        Optional context dictionary passed to the agent. Default ``None``.
    output_type
        Optional type used to cast the final output. Default ``None``.

    Returns
    -------
    Any
        Agent response, optionally converted to ``output_type``.
    """
    result = await Runner.run(agent, agent_input, context=agent_context)
    if output_type is not None:
        result = result.final_output_as(output_type)
    return result


def _run_agent_sync(
    agent: Agent,
    agent_input: str,
    agent_context: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """Run an ``Agent`` synchronously.

    Parameters
    ----------
    agent
        Configured agent instance to execute.
    agent_input
        Prompt or query string for the agent.
    agent_context
        Optional context dictionary passed to the agent. Default ``None``.

    Returns
    -------
    RunResult
        Result from the agent execution.
    """
    coro = Runner.run(agent, agent_input, context=agent_context)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop.is_running():
        result: RunResult | None = None

        def _thread_runner() -> None:
            nonlocal result
            result = asyncio.run(coro)

        thread = threading.Thread(target=_thread_runner, daemon=True)
        thread.start()
        thread.join()
        if result is None:
            raise RuntimeError("Agent execution did not return a result.")
        return result

    return loop.run_until_complete(coro)


def _run_agent_streamed(
    agent: Agent,
    agent_input: str,
    context: Optional[Dict[str, Any]] = None,
) -> RunResultStreaming:
    """Run an ``Agent`` synchronously and return a streaming result.

    Parameters
    ----------
    agent
        Configured agent to execute.
    agent_input
        Prompt or query string for the agent.
    context
        Optional context dictionary passed to the agent. Default ``None``.

    Returns
    -------
    RunResultStreaming
        Instance for streaming outputs.
    """
    result = Runner.run_streamed(agent, agent_input, context=context)
    return result


__all__ = [
    "AgentConfigLike",
    "BaseAgent",
    "_run_agent",
    "_run_agent_sync",
    "_run_agent_streamed",
]

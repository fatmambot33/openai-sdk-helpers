"""Convenience wrappers for running OpenAI agents.

These helpers provide a narrow surface around the lower-level functions in
``openai-sdk-helpers.agent.base`` so that callers can execute agents with consistent
signatures whether they need asynchronous, synchronous, or streamed results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents import Agent, RunResult, RunResultStreaming

from .base import _run_agent, _run_agent_streamed, _run_agent_sync


async def run_async(
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
    return await _run_agent(
        agent=agent,
        agent_input=agent_input,
        agent_context=agent_context,
        output_type=output_type,
    )


def run(
    agent: Agent,
    agent_input: str,
    agent_context: Optional[Dict[str, Any]] = None,
    output_type: Optional[Any] = None,
) -> Any:
    """Run an ``Agent`` synchronously.

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
    result: RunResult = _run_agent_sync(
        agent=agent,
        agent_input=agent_input,
        agent_context=agent_context,
    )
    if output_type:
        return result.final_output_as(output_type)
    return result


def run_streamed(
    agent: Agent,
    agent_input: str,
    agent_context: Optional[Dict[str, Any]] = None,
    output_type: Optional[Any] = None,
) -> RunResultStreaming:
    """Run an ``Agent`` and return a streaming result.

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
    RunResultStreaming
        Streaming output wrapper from the agent execution.
    """
    result = _run_agent_streamed(
        agent=agent,
        agent_input=agent_input,
        context=agent_context,
    )
    if output_type:
        return result.final_output_as(output_type)
    return result


__all__ = ["run", "run_async", "run_streamed"]

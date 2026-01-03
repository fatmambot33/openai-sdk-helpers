"""Convenience wrappers for running OpenAI agents.

These helpers provide a consistent interface around the lower-level functions in
the ``agent.base`` module, allowing callers to execute agents with consistent
signatures whether they need asynchronous, synchronous, or streamed results.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from agents import Agent, RunResult, RunResultStreaming, Runner

from openai_sdk_helpers.async_utils import run_coroutine_with_fallback


async def _run_async(
    agent: Agent,
    input: str,
    context: Optional[Dict[str, Any]] = None,
    output_type: Optional[Any] = None,
) -> Any:
    """Run an Agent asynchronously.

    Parameters
    ----------
    agent : Agent
        Configured agent instance to execute.
    input : str
        Prompt or query string for the agent.
    context : dict or None, default=None
        Optional context dictionary passed to the agent.
    output_type : type or None, default=None
        Optional type used to cast the final output.

    Returns
    -------
    Any
        Agent response, optionally converted to ``output_type``.
    """
    result = await Runner.run(agent, input, context=context)
    if output_type is not None:
        result = result.final_output_as(output_type)
    return result


def _run_sync(
    agent: Agent,
    input: str,
    context: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """Run an Agent synchronously.

    Parameters
    ----------
    agent : Agent
        Configured agent instance to execute.
    input : str
        Prompt or query string for the agent.
    context : dict or None, default=None
        Optional context dictionary passed to the agent.

    Returns
    -------
    RunResult
        Result from the agent execution.

    Raises
    ------
    AsyncExecutionError
        If execution fails or times out.
    """
    coro = Runner.run(agent, input, context=context)
    return run_coroutine_with_fallback(coro)


def _run_streamed(
    agent: Agent,
    input: str,
    context: Optional[Dict[str, Any]] = None,
) -> RunResultStreaming:
    """Stream agent execution results.

    Parameters
    ----------
    agent : Agent
        Configured agent to execute.
    input : str
        Prompt or query string for the agent.
    context : dict or None, default=None
        Optional context dictionary passed to the agent.

    Returns
    -------
    RunResultStreaming
        Instance for streaming outputs.
    """
    result = Runner.run_streamed(agent, input, context=context)
    return result


async def run_async(
    agent: Agent,
    input: str,
    context: Optional[Dict[str, Any]] = None,
    output_type: Optional[Any] = None,
) -> Any:
    """Run an Agent asynchronously.

    Parameters
    ----------
    agent : Agent
        Configured agent instance to execute.
    input : str
        Prompt or query string for the agent.
    context : dict or None, default=None
        Optional context dictionary passed to the agent.
    output_type : type or None, default=None
        Optional type used to cast the final output.

    Returns
    -------
    Any
        Agent response, optionally converted to ``output_type``.
    """
    return await _run_async(
        agent=agent,
        input=input,
        context=context,
        output_type=output_type,
    )


def run_sync(
    agent: Agent,
    input: str,
    context: Optional[Dict[str, Any]] = None,
    output_type: Optional[Any] = None,
) -> Any:
    """Run an Agent synchronously.

    Parameters
    ----------
    agent : Agent
        Configured agent instance to execute.
    input : str
        Prompt or query string for the agent.
    context : dict or None, default=None
        Optional context dictionary passed to the agent.
    output_type : type or None, default=None
        Optional type used to cast the final output.

    Returns
    -------
    Any
        Agent response, optionally converted to ``output_type``.
    """
    result: RunResult = _run_sync(
        agent=agent,
        input=input,
        context=context,
    )
    if output_type:
        return result.final_output_as(output_type)
    return result


def run_streamed(
    agent: Agent,
    input: str,
    context: Optional[Dict[str, Any]] = None,
    output_type: Optional[Any] = None,
) -> RunResultStreaming:
    """Stream agent execution results.

    Parameters
    ----------
    agent : Agent
        Configured agent instance to execute.
    input : str
        Prompt or query string for the agent.
    context : dict or None, default=None
        Optional context dictionary passed to the agent.
    output_type : type or None, default=None
        Optional type used to cast the final output.

    Returns
    -------
    RunResultStreaming
        Streaming output wrapper from the agent execution.
    """
    result = _run_streamed(
        agent=agent,
        input=input,
        context=context,
    )
    if output_type:
        return result.final_output_as(output_type)
    return result


__all__ = ["run_sync", "run_async", "run_streamed"]

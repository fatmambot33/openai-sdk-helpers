"""Convenience wrappers for running OpenAI agents.

These helpers provide a consistent interface around the lower-level functions in
the ``agent.base`` module, allowing callers to execute agents with consistent
signatures whether they need asynchronous, synchronous, or streamed results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents import Agent, RunResult, RunResultStreaming, Runner

from openai_sdk_helpers.async_utils import run_coroutine_with_fallback


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

    Examples
    --------
    >>> import asyncio
    >>> from agents import Agent
    >>> async def example():
    ...     agent = Agent(name="test", instructions="test", model="gpt-4o-mini")
    ...     result = await run_async(agent, "What is 2+2?")
    ...     return result
    >>> asyncio.run(example())  # doctest: +SKIP
    """
    result = await Runner.run(agent, input, context=context)
    if output_type is not None:
        return result.final_output_as(output_type)
    return result


def run_sync(
    agent: Agent,
    input: str,
    context: Optional[Dict[str, Any]] = None,
    output_type: Optional[Any] = None,
) -> Any:
    """Run an Agent synchronously.

    Internally uses async execution with proper event loop handling.
    If an event loop is already running, creates a new thread to avoid
    nested event loop errors.

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

    Raises
    ------
    AsyncExecutionError
        If execution fails or times out.

    Examples
    --------
    >>> from agents import Agent
    >>> agent = Agent(name="test", instructions="test", model="gpt-4o-mini")
    >>> result = run_sync(agent, "What is 2+2?")  # doctest: +SKIP
    """
    coro = Runner.run(agent, input, context=context)
    result: RunResult = run_coroutine_with_fallback(coro)
    if output_type is not None:
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
        Configured agent to execute.
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

    Examples
    --------
    >>> from agents import Agent
    >>> agent = Agent(name="test", instructions="test", model="gpt-4o-mini")
    >>> result = run_streamed(agent, "Explain AI")  # doctest: +SKIP
    >>> for chunk in result.stream_text():  # doctest: +SKIP
    ...     print(chunk, end="")
    """
    result = Runner.run_streamed(agent, input, context=context)
    if output_type is not None:
        return result.final_output_as(output_type)
    return result


__all__ = ["run_sync", "run_async", "run_streamed"]

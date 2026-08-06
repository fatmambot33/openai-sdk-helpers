"""Convenience wrappers for running OpenAI agents.

These helpers provide a consistent interface around the lower-level functions in
the ``agent.base`` module, allowing callers to execute agents with consistent
signatures whether they need asynchronous or synchronous results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

from agents import Agent, RunResult, Runner, Session

from openai_sdk_helpers.runtime import (
    OperationContext,
    run_observed_async,
    run_observed_sync,
)
from openai_sdk_helpers.state import AgentRunState, resolve_agent_state
from openai_sdk_helpers.utils.async_utils import run_coroutine_with_fallback

from ..structure.base import StructureBase


async def run_async(
    agent: Agent,
    input: str | list[dict[str, Any]],
    *,
    context: Optional[Dict[str, Any]] = None,
    output_structure: Optional[type[StructureBase]] = None,
    session: Optional[Session] = None,
    state: AgentRunState | None = None,
    operation_context: OperationContext | None = None,
) -> Any:
    """Run an Agent asynchronously.

    Parameters
    ----------
    agent : Agent
        Configured agent instance to execute.
    input : str or list[dict[str, Any]]
        Prompt text or structured input for the agent.
    context : dict or None, default=None
        Optional context dictionary passed to the agent.
    output_structure : type[StructureBase] or None, default=None
        Optional type used to cast the final output.
    session : Session or None, default=None
        Backward-compatible shorthand for ``AgentRunState(session=session)``.
    state : AgentRunState or None, default=None
        Explicit client-managed session or server-managed continuation choice.
    operation_context : OperationContext or None, default=None
        Optional explicit lifecycle and observability context.

    Returns
    -------
    Any
        Original agent result, optionally converted to ``output_structure``.

    Raises
    ------
    ValueError
        If session and server-managed state mechanisms are mixed.

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
    resolved_state = resolve_agent_state(state=state, session=session)

    async def execute() -> Any:
        result = await Runner.run(
            agent,
            cast(Any, input),
            context=context,
            session=resolved_state.session,
            previous_response_id=resolved_state.previous_response_id,
            auto_previous_response_id=resolved_state.auto_previous_response_id,
            conversation_id=resolved_state.conversation_id,
        )
        if output_structure is not None:
            return result.final_output_as(output_structure)
        return result

    return await run_observed_async(operation_context, execute)


def run_sync(
    agent: Agent,
    input: str | list[dict[str, Any]],
    *,
    context: Optional[Dict[str, Any]] = None,
    output_structure: Optional[type[StructureBase]] = None,
    session: Optional[Session] = None,
    state: AgentRunState | None = None,
    operation_context: OperationContext | None = None,
) -> Any:
    """Run an Agent synchronously.

    Internally uses async execution with proper event loop handling. If an event
    loop is already running, a new thread avoids nested event loop errors.

    Parameters
    ----------
    agent : Agent
        Configured agent instance to execute.
    input : str or list[dict[str, Any]]
        Prompt text or structured input for the agent.
    context : dict or None, default=None
        Optional context dictionary passed to the agent.
    output_structure : type[StructureBase] or None, default=None
        Optional type used to cast the final output.
    session : Session or None, default=None
        Backward-compatible shorthand for ``AgentRunState(session=session)``.
    state : AgentRunState or None, default=None
        Explicit client-managed session or server-managed continuation choice.
    operation_context : OperationContext or None, default=None
        Optional explicit lifecycle and observability context.

    Returns
    -------
    Any
        Original agent result, optionally converted to ``output_structure``.

    Raises
    ------
    ValueError
        If session and server-managed state mechanisms are mixed.
    AsyncExecutionError
        If execution fails or times out.

    Examples
    --------
    >>> from agents import Agent
    >>> agent = Agent(name="test", instructions="test", model="gpt-4o-mini")
    >>> result = run_sync(agent, "What is 2+2?")  # doctest: +SKIP
    """
    resolved_state = resolve_agent_state(state=state, session=session)

    def execute() -> Any:
        coro = Runner.run(
            agent,
            cast(Any, input),
            context=context,
            session=resolved_state.session,
            previous_response_id=resolved_state.previous_response_id,
            auto_previous_response_id=resolved_state.auto_previous_response_id,
            conversation_id=resolved_state.conversation_id,
        )
        result: RunResult = run_coroutine_with_fallback(coro)
        if output_structure is not None:
            return result.final_output_as(output_structure)
        return result

    return run_observed_sync(operation_context, execute)


__all__ = ["run_sync", "run_async"]

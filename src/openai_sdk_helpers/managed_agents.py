"""Thin helpers for the OpenAI-managed Agents API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal, cast

from .errors import OpenAISDKError
from .runtime import OperationContext, run_observed_async, run_observed_sync

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openai.pagination import AsyncCursorPage, SyncCursorPage
    from openai.types.beta.agent_session import AgentSession
    from openai.types.beta.agent_session_deleted import AgentSessionDeleted
    from openai.types.beta.agent_session_input_message_param import (
        AgentSessionInputMessageParam,
    )
    from openai.types.beta.agent_session_input_param import AgentSessionInputParam
    from openai.types.beta.agents import session_create_params
    from openai.types.beta.environment_param import EnvironmentParam

MIN_MANAGED_AGENTS_OPENAI_VERSION = "3.13.0"


class ManagedAgentsUnavailableError(OpenAISDKError):
    """The supplied OpenAI client does not expose the managed Agents API."""


def managed_agents_available(client: object) -> bool:
    """Return whether a client exposes the managed Agents API session resource.

    Parameters
    ----------
    client : object
        OpenAI SDK client or compatible test double.

    Returns
    -------
    bool
        ``True`` when ``client.beta.agents.sessions`` is available.
    """
    return _find_managed_agents_resource(client) is not None


class ManagedAgentsClient:
    """Synchronous facade for discrete managed-agent session operations.

    The facade preserves official SDK resources and results. Streaming,
    artifacts, subagents, items, turns, and other fast-moving beta surfaces stay
    available through :attr:`sessions` and :attr:`agents` rather than being
    reimplemented here.

    Parameters
    ----------
    sdk_client : OpenAI
        Configured synchronous OpenAI SDK client.

    Raises
    ------
    ManagedAgentsUnavailableError
        If the supplied client does not expose ``beta.agents.sessions``.
    """

    def __init__(self, sdk_client: OpenAI) -> None:
        """Initialize the facade and validate feature availability."""
        self._sdk_client = sdk_client
        self._agents = _require_managed_agents_resource(sdk_client)

    @property
    def sdk_client(self) -> OpenAI:
        """Return the underlying configured OpenAI SDK client."""
        return self._sdk_client

    @property
    def agents(self) -> Any:
        """Return the raw official ``client.beta.agents`` resource."""
        return self._agents

    @property
    def sessions(self) -> Any:
        """Return the raw official managed-agent sessions resource."""
        return self._agents.sessions

    def create_session(
        self,
        *,
        environment: EnvironmentParam,
        agent: session_create_params.Agent | None = None,
        agent_id: str | None = None,
        input: str | Iterable[AgentSessionInputMessageParam] | None = None,
        metadata: Mapping[str, str] | None = None,
        vault_ids: Sequence[str] | None = None,
        operation_context: OperationContext | None = None,
    ) -> AgentSession:
        """Create a non-streaming managed-agent session.

        Parameters
        ----------
        environment : EnvironmentParam
            Official SDK environment configuration or template reference.
        agent : session_create_params.Agent or None, default=None
            Inline agent configuration. Omitted when ``None``.
        agent_id : str or None, default=None
            Saved reusable agent identifier. Omitted when ``None``.
        input : str, Iterable[AgentSessionInputMessageParam], or None, default=None
            Optional initial input. Omitted when ``None``.
        metadata : Mapping[str, str] or None, default=None
            Optional session metadata. Omitted when ``None``.
        vault_ids : Sequence[str] or None, default=None
            Optional vault identifiers. Omitted when ``None``.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        AgentSession
            Original official SDK session object.

        Notes
        -----
        Use ``sessions.create(..., stream=True)`` or ``sessions.stream(...)``
        directly for streaming so the official stream lifecycle remains intact.
        """
        kwargs: dict[str, Any] = {"environment": environment}
        if agent is not None:
            kwargs["agent"] = agent
        if agent_id is not None:
            kwargs["agent_id"] = agent_id
        if input is not None:
            kwargs["input"] = input
        if metadata is not None:
            kwargs["metadata"] = dict(metadata)
        if vault_ids is not None:
            kwargs["vault_ids"] = list(vault_ids)
        return run_observed_sync(
            operation_context,
            lambda: cast("AgentSession", self.sessions.create(**kwargs)),
        )

    def retrieve_session(
        self,
        session_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> AgentSession:
        """Retrieve one managed-agent session by identifier."""
        session_id = _required_session_id(session_id)
        return run_observed_sync(
            operation_context,
            lambda: cast("AgentSession", self.sessions.retrieve(session_id)),
        )

    def update_session_metadata(
        self,
        session_id: str,
        *,
        metadata: Mapping[str, str] | None,
        operation_context: OperationContext | None = None,
    ) -> AgentSession:
        """Replace or clear metadata for one managed-agent session."""
        session_id = _required_session_id(session_id)
        normalized_metadata = None if metadata is None else dict(metadata)
        return run_observed_sync(
            operation_context,
            lambda: cast(
                "AgentSession",
                self.sessions.update(session_id, metadata=normalized_metadata),
            ),
        )

    def list_sessions(
        self,
        *,
        after: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
        order: Literal["asc", "desc"] | None = None,
        operation_context: OperationContext | None = None,
    ) -> SyncCursorPage[AgentSession]:
        """List managed-agent sessions using official cursor pagination."""
        kwargs: dict[str, Any] = {}
        if after is not None:
            kwargs["after"] = after
        if agent_id is not None:
            kwargs["agent_id"] = agent_id
        if limit is not None:
            kwargs["limit"] = limit
        if order is not None:
            kwargs["order"] = order
        return run_observed_sync(
            operation_context,
            lambda: cast(
                "SyncCursorPage[AgentSession]",
                self.sessions.list(**kwargs),
            ),
        )

    def delete_session(
        self,
        session_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> AgentSessionDeleted:
        """Delete one managed-agent session and return SDK confirmation."""
        session_id = _required_session_id(session_id)
        return run_observed_sync(
            operation_context,
            lambda: cast("AgentSessionDeleted", self.sessions.delete(session_id)),
        )

    def submit_events(
        self,
        session_id: str,
        *,
        events: Iterable[AgentSessionInputParam],
        idempotency_key: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> None:
        """Submit message, cancellation, or tool-result events to a session."""
        session_id = _required_session_id(session_id)
        kwargs: dict[str, Any] = {"events": events}
        if idempotency_key is not None:
            kwargs["idempotency_key"] = idempotency_key
        return run_observed_sync(
            operation_context,
            lambda: self.sessions.events.create(session_id, **kwargs),
        )


class AsyncManagedAgentsClient:
    """Asynchronous facade for discrete managed-agent session operations.

    Parameters
    ----------
    sdk_client : AsyncOpenAI
        Configured asynchronous OpenAI SDK client.

    Raises
    ------
    ManagedAgentsUnavailableError
        If the supplied client does not expose ``beta.agents.sessions``.
    """

    def __init__(self, sdk_client: AsyncOpenAI) -> None:
        """Initialize the facade and validate feature availability."""
        self._sdk_client = sdk_client
        self._agents = _require_managed_agents_resource(sdk_client)

    @property
    def sdk_client(self) -> AsyncOpenAI:
        """Return the underlying configured asynchronous OpenAI SDK client."""
        return self._sdk_client

    @property
    def agents(self) -> Any:
        """Return the raw official ``client.beta.agents`` resource."""
        return self._agents

    @property
    def sessions(self) -> Any:
        """Return the raw official asynchronous sessions resource."""
        return self._agents.sessions

    async def create_session(
        self,
        *,
        environment: EnvironmentParam,
        agent: session_create_params.Agent | None = None,
        agent_id: str | None = None,
        input: str | Iterable[AgentSessionInputMessageParam] | None = None,
        metadata: Mapping[str, str] | None = None,
        vault_ids: Sequence[str] | None = None,
        operation_context: OperationContext | None = None,
    ) -> AgentSession:
        """Create a non-streaming managed-agent session asynchronously."""
        kwargs: dict[str, Any] = {"environment": environment}
        if agent is not None:
            kwargs["agent"] = agent
        if agent_id is not None:
            kwargs["agent_id"] = agent_id
        if input is not None:
            kwargs["input"] = input
        if metadata is not None:
            kwargs["metadata"] = dict(metadata)
        if vault_ids is not None:
            kwargs["vault_ids"] = list(vault_ids)
        return await run_observed_async(
            operation_context,
            lambda: self.sessions.create(**kwargs),
        )

    async def retrieve_session(
        self,
        session_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> AgentSession:
        """Retrieve one managed-agent session asynchronously."""
        session_id = _required_session_id(session_id)
        return await run_observed_async(
            operation_context,
            lambda: self.sessions.retrieve(session_id),
        )

    async def update_session_metadata(
        self,
        session_id: str,
        *,
        metadata: Mapping[str, str] | None,
        operation_context: OperationContext | None = None,
    ) -> AgentSession:
        """Replace or clear session metadata asynchronously."""
        session_id = _required_session_id(session_id)
        normalized_metadata = None if metadata is None else dict(metadata)
        return await run_observed_async(
            operation_context,
            lambda: self.sessions.update(
                session_id,
                metadata=normalized_metadata,
            ),
        )

    async def list_sessions(
        self,
        *,
        after: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
        order: Literal["asc", "desc"] | None = None,
        operation_context: OperationContext | None = None,
    ) -> AsyncCursorPage[AgentSession]:
        """List managed-agent sessions asynchronously."""
        kwargs: dict[str, Any] = {}
        if after is not None:
            kwargs["after"] = after
        if agent_id is not None:
            kwargs["agent_id"] = agent_id
        if limit is not None:
            kwargs["limit"] = limit
        if order is not None:
            kwargs["order"] = order
        return await run_observed_async(
            operation_context,
            lambda: self.sessions.list(**kwargs),
        )

    async def delete_session(
        self,
        session_id: str,
        *,
        operation_context: OperationContext | None = None,
    ) -> AgentSessionDeleted:
        """Delete one managed-agent session asynchronously."""
        session_id = _required_session_id(session_id)
        return await run_observed_async(
            operation_context,
            lambda: self.sessions.delete(session_id),
        )

    async def submit_events(
        self,
        session_id: str,
        *,
        events: Iterable[AgentSessionInputParam],
        idempotency_key: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> None:
        """Submit input events to a managed-agent session asynchronously."""
        session_id = _required_session_id(session_id)
        kwargs: dict[str, Any] = {"events": events}
        if idempotency_key is not None:
            kwargs["idempotency_key"] = idempotency_key
        return await run_observed_async(
            operation_context,
            lambda: self.sessions.events.create(session_id, **kwargs),
        )


def _find_managed_agents_resource(client: object) -> Any | None:
    beta = getattr(client, "beta", None)
    agents = getattr(beta, "agents", None) if beta is not None else None
    sessions = getattr(agents, "sessions", None) if agents is not None else None
    if sessions is None:
        return None
    return agents


def _require_managed_agents_resource(client: object) -> Any:
    agents = _find_managed_agents_resource(client)
    if agents is not None:
        return agents
    installed_version = _installed_openai_version()
    raise ManagedAgentsUnavailableError(
        "Managed Agents API helpers require a client exposing "
        "client.beta.agents.sessions (available in openai>="
        f"{MIN_MANAGED_AGENTS_OPENAI_VERSION}). Installed openai version: "
        f"{installed_version}. Upgrade openai or use another supported helper surface.",
        context={
            "feature": "managed_agents",
            "minimum_openai_version": MIN_MANAGED_AGENTS_OPENAI_VERSION,
            "installed_openai_version": installed_version,
        },
    )


def _installed_openai_version() -> str:
    try:
        return version("openai")
    except PackageNotFoundError:
        return "not installed"


def _required_session_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("session_id must not be empty")
    return normalized


__all__ = [
    "AsyncManagedAgentsClient",
    "MIN_MANAGED_AGENTS_OPENAI_VERSION",
    "ManagedAgentsClient",
    "ManagedAgentsUnavailableError",
    "managed_agents_available",
]

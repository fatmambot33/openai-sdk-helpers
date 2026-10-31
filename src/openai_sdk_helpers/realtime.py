"""Thin helpers for OpenAI Realtime API sessions and events.

The module keeps transport, audio devices, and the full event protocol in the
official OpenAI SDK. It only covers repeated server-side session plumbing,
normalizes the event envelope, and makes cancellation and tool-result handling
explicit.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from .errors import OpenAISDKError
from .runtime import OperationContext, run_observed_async, run_observed_sync

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI


MIN_REALTIME_OPENAI_VERSION = "3.41.0"


class RealtimeUnavailableError(OpenAISDKError):
    """The supplied OpenAI client does not expose the Realtime API."""


@dataclass(frozen=True)
class RealtimeEvent:
    """Expose an event type while preserving the original SDK event object.

    Parameters
    ----------
    type : str
        Official Realtime event type.
    raw : object
        Unchanged official SDK event object or mapping.
    """

    type: str
    raw: object


def realtime_available(client: object) -> bool:
    """Return whether a client exposes the Realtime API resources used here.

    Parameters
    ----------
    client : object
        OpenAI SDK client or compatible test double.

    Returns
    -------
    bool
        ``True`` when ``client.realtime`` exposes ``connect`` and
        ``client_secrets.create``.
    """
    return _find_realtime_resource(client) is not None


def normalize_realtime_event(event: object) -> RealtimeEvent:
    """Return a shallow normalized view over one official Realtime event.

    Parameters
    ----------
    event : object
        Official SDK event model or a mapping returned by a compatible client.

    Returns
    -------
    RealtimeEvent
        Event type plus the unchanged original event.

    Raises
    ------
    ValueError
        If the event has no non-empty string ``type`` field.
    """
    event_type = (
        event.get("type")
        if isinstance(event, Mapping)
        else getattr(event, "type", None)
    )
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("Realtime events must expose a non-empty string `type` field.")
    return RealtimeEvent(type=event_type, raw=event)


class RealtimeClient:
    """Provide synchronous helpers for official Realtime sessions and events.

    The facade preserves the official connection manager, connection, resources,
    client-secret response, server events, and exceptions. It does not manage a
    browser, audio device, reconnection loop, or alternate event protocol.

    Parameters
    ----------
    sdk_client : OpenAI
        Configured synchronous OpenAI SDK client.

    Raises
    ------
    RealtimeUnavailableError
        If the supplied client does not expose ``realtime``.

    Methods
    -------
    create_client_secret
        Create an official Realtime client-secret session response.
    connect
        Return the official Realtime connection manager.
    update_session
        Send an official ``session.update`` event.
    iter_events
        Yield normalized views over server events.
    cancel_response
        Send an explicit ``response.cancel`` event.
    submit_tool_result
        Add a function-call output and optionally request the next response.
    """

    def __init__(self, sdk_client: OpenAI) -> None:
        """Initialize the facade and validate feature availability."""
        self._sdk_client = sdk_client
        self._realtime = _require_realtime_resource(sdk_client)

    @property
    def sdk_client(self) -> OpenAI:
        """Return the configured official OpenAI client."""
        return self._sdk_client

    @property
    def realtime(self) -> Any:
        """Return the raw official ``client.realtime`` resource."""
        return self._realtime

    @property
    def client_secrets(self) -> Any:
        """Return the raw official Realtime client-secrets resource."""
        return self._realtime.client_secrets

    def create_client_secret(
        self,
        *,
        session: Mapping[str, Any] | None = None,
        operation_context: OperationContext | None = None,
    ) -> Any:
        """Create an official ephemeral client-secret session response.

        Parameters
        ----------
        session : Mapping[str, Any] or None, default=None
            Official Realtime session configuration forwarded unchanged as the
            native ``session`` parameter. Omit to use the upstream defaults.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for the HTTP request.

        Returns
        -------
        Any
            Original official SDK client-secret session response.

        Notes
        -----
        Treat the returned client secret as a credential. The helper does not log,
        store, or expose it beyond the original SDK response.
        """
        kwargs: dict[str, Any] = {}
        if session is not None:
            kwargs["session"] = dict(session)
        return run_observed_sync(
            operation_context,
            lambda: self.client_secrets.create(**kwargs),
        )

    def connect(
        self,
        *,
        model: str,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        websocket_connection_options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return the official synchronous Realtime connection manager.

        Parameters
        ----------
        model : str
            Realtime model identifier accepted by the installed SDK.
        extra_headers : Mapping[str, str] or None, default=None
            Additional websocket handshake headers.
        extra_query : Mapping[str, str] or None, default=None
            Additional websocket query parameters.
        websocket_connection_options : Mapping[str, Any] or None, default=None
            Low-level official SDK websocket options.

        Returns
        -------
        Any
            Official SDK connection manager. Enter it with ``with`` so the SDK
            owns connection cleanup.
        """
        return self.realtime.connect(
            model=model,
            extra_headers=dict(extra_headers or {}),
            extra_query=dict(extra_query or {}),
            websocket_connection_options=dict(websocket_connection_options or {}),
        )

    def update_session(
        self,
        connection: Any,
        *,
        session: Mapping[str, Any],
        event_id: str | None = None,
    ) -> None:
        """Send an official ``session.update`` event through a live connection.

        Parameters
        ----------
        connection : Any
            Active official Realtime connection.
        session : Mapping[str, Any]
            Partial official Realtime session configuration.
        event_id : str or None, default=None
            Optional caller-owned event identifier.
        """
        kwargs: dict[str, Any] = {"session": dict(session)}
        if event_id is not None:
            kwargs["event_id"] = event_id
        connection.session.update(**kwargs)

    def iter_events(self, connection: Iterable[object]) -> Iterator[RealtimeEvent]:
        """Yield normalized event envelopes from an official connection iterator.

        Parameters
        ----------
        connection : Iterable[object]
            Active official Realtime connection or another event iterable.

        Yields
        ------
        RealtimeEvent
            Shallow view with the official event type and unchanged raw event.
        """
        for event in connection:
            yield normalize_realtime_event(event)

    def cancel_response(
        self,
        connection: Any,
        *,
        response_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """Send an explicit official ``response.cancel`` event.

        Parameters
        ----------
        connection : Any
            Active official Realtime connection.
        response_id : str or None, default=None
            Optional in-progress response identifier.
        event_id : str or None, default=None
            Optional caller-owned event identifier.
        """
        kwargs: dict[str, Any] = {}
        if response_id is not None:
            kwargs["response_id"] = response_id
        if event_id is not None:
            kwargs["event_id"] = event_id
        connection.response.cancel(**kwargs)

    def submit_tool_result(
        self,
        connection: Any,
        *,
        call_id: str,
        output: str,
        create_response: bool = True,
        event_id: str | None = None,
    ) -> None:
        """Submit function output and optionally request the next model response.

        Parameters
        ----------
        connection : Any
            Active official Realtime connection.
        call_id : str
            Official function-call identifier to resolve.
        output : str
            Serialized function result accepted by the official SDK.
        create_response : bool, default=True
            Request a new response after creating the function output item.
        event_id : str or None, default=None
            Optional caller-owned identifier for the conversation-item event.
        """
        item_kwargs: dict[str, Any] = {
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        }
        if event_id is not None:
            item_kwargs["event_id"] = event_id
        connection.conversation.item.create(**item_kwargs)
        if create_response:
            connection.response.create()


class AsyncRealtimeClient:
    """Provide asynchronous helpers for official Realtime sessions and events.

    Parameters
    ----------
    sdk_client : AsyncOpenAI
        Configured asynchronous OpenAI SDK client.

    Raises
    ------
    RealtimeUnavailableError
        If the supplied client does not expose ``realtime``.

    Methods
    -------
    create_client_secret
        Create an official Realtime client-secret session response asynchronously.
    connect
        Return the official asynchronous connection manager.
    update_session
        Send an official ``session.update`` event asynchronously.
    iter_events
        Yield normalized views over asynchronous server events.
    cancel_response
        Send an explicit ``response.cancel`` event asynchronously.
    submit_tool_result
        Add a function-call output and optionally request the next response.
    """

    def __init__(self, sdk_client: AsyncOpenAI) -> None:
        """Initialize the facade and validate feature availability."""
        self._sdk_client = sdk_client
        self._realtime = _require_realtime_resource(sdk_client)

    @property
    def sdk_client(self) -> AsyncOpenAI:
        """Return the configured official asynchronous OpenAI client."""
        return self._sdk_client

    @property
    def realtime(self) -> Any:
        """Return the raw official asynchronous Realtime resource."""
        return self._realtime

    @property
    def client_secrets(self) -> Any:
        """Return the raw official asynchronous client-secrets resource."""
        return self._realtime.client_secrets

    async def create_client_secret(
        self,
        *,
        session: Mapping[str, Any] | None = None,
        operation_context: OperationContext | None = None,
    ) -> Any:
        """Create an official ephemeral client-secret response asynchronously.

        Parameters
        ----------
        session : Mapping[str, Any] or None, default=None
            Official Realtime session configuration forwarded unchanged as the
            native ``session`` parameter. Omit to use the upstream defaults.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for the HTTP request.

        Returns
        -------
        Any
            Original official SDK client-secret session response.
        """
        kwargs: dict[str, Any] = {}
        if session is not None:
            kwargs["session"] = dict(session)
        return await run_observed_async(
            operation_context,
            lambda: self.client_secrets.create(**kwargs),
        )

    def connect(
        self,
        *,
        model: str,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        websocket_connection_options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return the official asynchronous Realtime connection manager.

        Parameters
        ----------
        model : str
            Realtime model identifier accepted by the installed SDK.
        extra_headers : Mapping[str, str] or None, default=None
            Additional websocket handshake headers.
        extra_query : Mapping[str, str] or None, default=None
            Additional websocket query parameters.
        websocket_connection_options : Mapping[str, Any] or None, default=None
            Low-level official SDK websocket options.

        Returns
        -------
        Any
            Official SDK asynchronous connection manager. Enter it with ``async
            with`` so the SDK owns connection cleanup.
        """
        return self.realtime.connect(
            model=model,
            extra_headers=dict(extra_headers or {}),
            extra_query=dict(extra_query or {}),
            websocket_connection_options=dict(websocket_connection_options or {}),
        )

    async def update_session(
        self,
        connection: Any,
        *,
        session: Mapping[str, Any],
        event_id: str | None = None,
    ) -> None:
        """Send an official ``session.update`` event asynchronously.

        Parameters
        ----------
        connection : Any
            Active official asynchronous Realtime connection.
        session : Mapping[str, Any]
            Partial official Realtime session configuration.
        event_id : str or None, default=None
            Optional caller-owned event identifier.
        """
        kwargs: dict[str, Any] = {"session": dict(session)}
        if event_id is not None:
            kwargs["event_id"] = event_id
        await connection.session.update(**kwargs)

    async def iter_events(
        self,
        connection: AsyncIterable[object],
    ) -> AsyncIterator[RealtimeEvent]:
        """Yield normalized event envelopes from an official async iterator.

        Parameters
        ----------
        connection : AsyncIterable[object]
            Active official asynchronous Realtime connection or event iterable.

        Yields
        ------
        RealtimeEvent
            Shallow view with the official event type and unchanged raw event.
        """
        async for event in connection:
            yield normalize_realtime_event(event)

    async def cancel_response(
        self,
        connection: Any,
        *,
        response_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """Send an explicit official ``response.cancel`` event asynchronously.

        Parameters
        ----------
        connection : Any
            Active official asynchronous Realtime connection.
        response_id : str or None, default=None
            Optional in-progress response identifier.
        event_id : str or None, default=None
            Optional caller-owned event identifier.
        """
        kwargs: dict[str, Any] = {}
        if response_id is not None:
            kwargs["response_id"] = response_id
        if event_id is not None:
            kwargs["event_id"] = event_id
        await connection.response.cancel(**kwargs)

    async def submit_tool_result(
        self,
        connection: Any,
        *,
        call_id: str,
        output: str,
        create_response: bool = True,
        event_id: str | None = None,
    ) -> None:
        """Submit function output and optionally request the next model response.

        Parameters
        ----------
        connection : Any
            Active official asynchronous Realtime connection.
        call_id : str
            Official function-call identifier to resolve.
        output : str
            Serialized function result accepted by the official SDK.
        create_response : bool, default=True
            Request a new response after creating the function output item.
        event_id : str or None, default=None
            Optional caller-owned identifier for the conversation-item event.
        """
        item_kwargs: dict[str, Any] = {
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        }
        if event_id is not None:
            item_kwargs["event_id"] = event_id
        await connection.conversation.item.create(**item_kwargs)
        if create_response:
            await connection.response.create()


def _find_realtime_resource(client: object) -> Any | None:
    """Return the Realtime resource when its required methods are available."""
    realtime = getattr(client, "realtime", None)
    client_secrets = getattr(realtime, "client_secrets", None)
    if callable(getattr(realtime, "connect", None)) and callable(
        getattr(client_secrets, "create", None)
    ):
        return realtime
    return None


def _require_realtime_resource(client: object) -> Any:
    """Return the supported Realtime resource or raise an actionable error."""
    realtime = _find_realtime_resource(client)
    if realtime is None:
        raise RealtimeUnavailableError(
            "The supplied client does not expose `realtime.connect` and "
            "`realtime.client_secrets.create`. Install `openai[realtime]>=3.41.0` "
            "and inject a compatible OpenAI client.",
            context={
                "feature": "realtime",
                "minimum_openai_version": MIN_REALTIME_OPENAI_VERSION,
            },
        )
    return realtime


__all__ = [
    "AsyncRealtimeClient",
    "MIN_REALTIME_OPENAI_VERSION",
    "RealtimeClient",
    "RealtimeEvent",
    "RealtimeUnavailableError",
    "normalize_realtime_event",
    "realtime_available",
]

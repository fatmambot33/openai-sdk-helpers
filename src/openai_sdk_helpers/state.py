"""Explicit conversation state ownership and persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .response.messages import ResponseMessages

if TYPE_CHECKING:
    from agents import Session


class ConversationStateMode(str, Enum):
    """Supported conversation state ownership modes."""

    STATELESS = "stateless"
    APPLICATION_HISTORY = "application_history"
    PACKAGE_LOCAL = "package_local"
    AGENT_SESSION = "agent_session"
    PREVIOUS_RESPONSE = "previous_response"
    CONVERSATION = "conversation"


@dataclass(frozen=True, slots=True)
class ResponseContinuation:
    """Server-managed Responses continuation identifiers.

    Parameters
    ----------
    previous_response_id : str or None, default=None
        Previous Responses API response identifier.
    conversation_id : str or None, default=None
        OpenAI Conversations API identifier.

    Notes
    -----
    The two identifiers are mutually exclusive. This object does not make an API
    call, store history, or automatically select a mechanism.
    """

    previous_response_id: str | None = None
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize identifiers and reject ambiguous continuation."""
        previous_response_id = _normalized_identifier(
            self.previous_response_id,
            "previous_response_id",
        )
        conversation_id = _normalized_identifier(
            self.conversation_id,
            "conversation_id",
        )
        if previous_response_id is not None and conversation_id is not None:
            raise ValueError(
                "previous_response_id and conversation_id are mutually exclusive"
            )
        object.__setattr__(self, "previous_response_id", previous_response_id)
        object.__setattr__(self, "conversation_id", conversation_id)

    @property
    def mode(self) -> ConversationStateMode:
        """Return the selected server-managed state mode."""
        if self.previous_response_id is not None:
            return ConversationStateMode.PREVIOUS_RESPONSE
        if self.conversation_id is not None:
            return ConversationStateMode.CONVERSATION
        return ConversationStateMode.STATELESS

    def apply(self, request_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Copy request keyword arguments and add the selected identifier.

        Parameters
        ----------
        request_kwargs : Mapping[str, Any]
            Existing official SDK request keyword arguments.

        Returns
        -------
        dict[str, Any]
            Copied keyword arguments with at most one continuation identifier.

        Raises
        ------
        ValueError
            If the input mapping already contains a conflicting identifier.
        """
        resolved = dict(request_kwargs)
        existing_previous = resolved.get("previous_response_id")
        existing_conversation = resolved.get("conversation")
        if existing_conversation is None:
            existing_conversation = resolved.get("conversation_id")
        if self.previous_response_id is not None:
            if existing_conversation is not None:
                raise ValueError(
                    "conversation continuation cannot be combined with "
                    "previous_response_id"
                )
            if existing_previous not in (None, self.previous_response_id):
                raise ValueError("request already has a different previous_response_id")
            resolved["previous_response_id"] = self.previous_response_id
        if self.conversation_id is not None:
            if existing_previous is not None:
                raise ValueError(
                    "previous_response_id cannot be combined with conversation"
                )
            if existing_conversation not in (None, self.conversation_id):
                raise ValueError("request already has a different conversation identifier")
            resolved["conversation"] = self.conversation_id
        return resolved


@dataclass(frozen=True, slots=True)
class AgentRunState:
    """Explicit state selection for one Agents SDK run.

    Parameters
    ----------
    session : Session or None, default=None
        Underlying client-managed Agents SDK session.
    previous_response_id : str or None, default=None
        Previous Responses API response identifier.
    conversation_id : str or None, default=None
        OpenAI Conversations API identifier.
    auto_previous_response_id : bool, default=False
        Enable Agents SDK automatic response chaining.

    Notes
    -----
    A session cannot be combined with any server-managed setting.
    ``conversation_id`` cannot be combined with either previous-response option.
    ``previous_response_id`` may be combined with
    ``auto_previous_response_id=True`` as supported by the Agents SDK.
    """

    session: Session | None = None
    previous_response_id: str | None = None
    conversation_id: str | None = None
    auto_previous_response_id: bool = False

    def __post_init__(self) -> None:
        """Normalize identifiers and reject mixed state ownership."""
        previous_response_id = _normalized_identifier(
            self.previous_response_id,
            "previous_response_id",
        )
        conversation_id = _normalized_identifier(
            self.conversation_id,
            "conversation_id",
        )
        server_managed = (
            previous_response_id is not None
            or conversation_id is not None
            or self.auto_previous_response_id
        )
        if self.session is not None and server_managed:
            raise ValueError(
                "Agents SDK session cannot be combined with conversation_id, "
                "previous_response_id, or auto_previous_response_id"
            )
        if conversation_id is not None and (
            previous_response_id is not None or self.auto_previous_response_id
        ):
            raise ValueError(
                "conversation_id cannot be combined with previous_response_id "
                "or auto_previous_response_id"
            )
        object.__setattr__(self, "previous_response_id", previous_response_id)
        object.__setattr__(self, "conversation_id", conversation_id)

    @property
    def mode(self) -> ConversationStateMode:
        """Return the selected state ownership mode."""
        if self.session is not None:
            return ConversationStateMode.AGENT_SESSION
        if self.conversation_id is not None:
            return ConversationStateMode.CONVERSATION
        if self.previous_response_id is not None or self.auto_previous_response_id:
            return ConversationStateMode.PREVIOUS_RESPONSE
        return ConversationStateMode.STATELESS


@dataclass(frozen=True, slots=True)
class LocalMessageStore:
    """Explicit package-local storage for ``ResponseMessages``.

    Parameters
    ----------
    path : Path or str
        JSON file owned by the caller.

    Notes
    -----
    Construction, loading, and closing do not save automatically. Call ``save``
    explicitly. ``clear`` writes an empty message collection, while ``delete``
    removes the caller-owned file.
    """

    path: Path | str

    def __post_init__(self) -> None:
        """Normalize the caller-owned storage path."""
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())

    def save(self, messages: ResponseMessages) -> Path:
        """Persist messages to the configured JSON file.

        Parameters
        ----------
        messages : ResponseMessages
            Message collection to serialize.

        Returns
        -------
        Path
            Resolved storage path.
        """
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        messages.to_json_file(str(path))
        return path

    def resume(self) -> ResponseMessages:
        """Load and return the persisted message collection.

        Returns
        -------
        ResponseMessages
            Restored messages.

        Raises
        ------
        FileNotFoundError
            If no saved state exists.
        """
        return ResponseMessages.from_json_file(str(self._path))

    def clear(self) -> ResponseMessages:
        """Replace persisted history with an empty message collection.

        Returns
        -------
        ResponseMessages
            Empty collection written to storage.
        """
        messages = ResponseMessages()
        self.save(messages)
        return messages

    def delete(self, *, missing_ok: bool = True) -> bool:
        """Delete the caller-owned storage file.

        Parameters
        ----------
        missing_ok : bool, default=True
            Return ``False`` when the file is absent instead of raising.

        Returns
        -------
        bool
            ``True`` when a file was removed and ``False`` when absent.

        Raises
        ------
        FileNotFoundError
            If the file is absent and ``missing_ok`` is ``False``.
        """
        path = self._path
        if not path.exists():
            if missing_ok:
                return False
            raise FileNotFoundError(path)
        path.unlink()
        return True

    @property
    def exists(self) -> bool:
        """Return whether saved local state exists."""
        return self._path.is_file()

    @property
    def _path(self) -> Path:
        return self.path if isinstance(self.path, Path) else Path(self.path)


def resolve_agent_state(
    *,
    state: AgentRunState | None,
    session: Session | None,
) -> AgentRunState:
    """Resolve the new state object and legacy ``session`` shorthand.

    Parameters
    ----------
    state : AgentRunState or None
        Explicit state selection.
    session : Session or None
        Backward-compatible session parameter.

    Returns
    -------
    AgentRunState
        Validated state object.

    Raises
    ------
    ValueError
        If both forms are supplied.
    """
    if state is not None and session is not None:
        raise ValueError("Pass either state or session, not both")
    if state is not None:
        return state
    return AgentRunState(session=session)


def _normalized_identifier(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


__all__ = [
    "AgentRunState",
    "ConversationStateMode",
    "LocalMessageStore",
    "ResponseContinuation",
    "resolve_agent_state",
]

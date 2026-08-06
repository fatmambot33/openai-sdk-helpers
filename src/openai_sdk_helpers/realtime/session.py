"""Typed configuration and explicit lifecycle for official Realtime sessions."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar, cast

T = TypeVar("T")

from openai_sdk_helpers.runtime import OperationContext, run_observed_async


class RealtimeLifecycleState(str, Enum):
    """Lifecycle states for one managed Realtime session."""

    CREATED = "created"
    STARTING = "starting"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RealtimeSessionConfig:
    """Typed server-side Realtime model and session settings.

    Parameters
    ----------
    model : str or None, default=None
        Realtime model identifier.
    voice : str or None, default=None
        Server-side voice identifier.
    instructions : str or None, default=None
        Session-level instructions.
    modalities : tuple[str, ...], default=()
        Explicit modalities such as ``audio`` and ``text``.
    input_audio_format : str or None, default=None
        Official input audio format value.
    output_audio_format : str or None, default=None
        Official output audio format value.
    turn_detection : Mapping[str, Any] or None, default=None
        Official turn-detection configuration.
    input_audio_transcription : Mapping[str, Any] or None, default=None
        Official transcription configuration.
    input_audio_noise_reduction : Mapping[str, Any] or None, default=None
        Official noise-reduction configuration.
    tool_choice : str or Mapping[str, Any] or None, default=None
        Official tool-choice configuration.
    extra : Mapping[str, Any], default={}
        Additional official SDK settings preserved without renaming.
    """

    model: str | None = None
    voice: str | None = None
    instructions: str | None = None
    modalities: tuple[str, ...] = ()
    input_audio_format: str | None = None
    output_audio_format: str | None = None
    turn_detection: Mapping[str, Any] | None = None
    input_audio_transcription: Mapping[str, Any] | None = None
    input_audio_noise_reduction: Mapping[str, Any] | None = None
    tool_choice: str | Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize text values and copy mutable settings."""
        for name in (
            "model",
            "voice",
            "instructions",
            "input_audio_format",
            "output_audio_format",
        ):
            value = getattr(self, name)
            if value is not None:
                normalized = value.strip()
                if not normalized:
                    raise ValueError(f"{name} must not be empty")
                object.__setattr__(self, name, normalized)
        modalities = tuple(
            dict.fromkeys(_required(value, "modality") for value in self.modalities)
        )
        object.__setattr__(self, "modalities", modalities)
        for name in (
            "turn_detection",
            "input_audio_transcription",
            "input_audio_noise_reduction",
            "extra",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, dict(value))
        if isinstance(self.tool_choice, Mapping):
            object.__setattr__(self, "tool_choice", dict(self.tool_choice))
        elif isinstance(self.tool_choice, str):
            object.__setattr__(
                self,
                "tool_choice",
                _required(self.tool_choice, "tool_choice"),
            )

    def as_model_config(self) -> dict[str, Any]:
        """Return an SDK-shaped copied model/session configuration."""
        config = dict(self.extra)
        values: dict[str, object | None] = {
            "model": self.model,
            "voice": self.voice,
            "instructions": self.instructions,
            "modalities": list(self.modalities) if self.modalities else None,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "turn_detection": (
                dict(self.turn_detection) if self.turn_detection is not None else None
            ),
            "input_audio_transcription": (
                dict(self.input_audio_transcription)
                if self.input_audio_transcription is not None
                else None
            ),
            "input_audio_noise_reduction": (
                dict(self.input_audio_noise_reduction)
                if self.input_audio_noise_reduction is not None
                else None
            ),
            "tool_choice": (
                dict(self.tool_choice)
                if isinstance(self.tool_choice, Mapping)
                else self.tool_choice
            ),
        }
        config.update(
            {name: value for name, value in values.items() if value is not None}
        )
        return config


@dataclass(frozen=True, slots=True)
class RealtimeRunnerConfig:
    """Official runner configuration with explicit tracing metadata.

    Parameters
    ----------
    session : RealtimeSessionConfig or None, default=None
        Typed model/session settings.
    workflow_name : str or None, default=None
        Optional official trace workflow name.
    group_id : str or None, default=None
        Optional trace grouping identifier.
    trace_metadata : Mapping[str, Any], default={}
        Copied trace metadata. Sensitive content must not be stored here.
    tracing_disabled : bool, default=False
        Disable official Agents SDK tracing explicitly.
    extra : Mapping[str, Any], default={}
        Additional official runner configuration.
    """

    session: RealtimeSessionConfig | None = None
    workflow_name: str | None = None
    group_id: str | None = None
    trace_metadata: Mapping[str, Any] = field(default_factory=dict)
    tracing_disabled: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize runner identity and copy mappings."""
        for name in ("workflow_name", "group_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required(value, name))
        object.__setattr__(self, "trace_metadata", dict(self.trace_metadata))
        object.__setattr__(self, "extra", dict(self.extra))

    def as_sdk_config(self) -> dict[str, Any]:
        """Return copied configuration for the official Realtime runner."""
        config = dict(self.extra)
        if self.session is not None:
            config["model_settings"] = self.session.as_model_config()
        if self.workflow_name is not None:
            config["workflow_name"] = self.workflow_name
        if self.group_id is not None:
            config["group_id"] = self.group_id
        if self.trace_metadata:
            config["trace_metadata"] = dict(self.trace_metadata)
        if self.tracing_disabled:
            config["tracing_disabled"] = True
        return config


@dataclass(frozen=True, slots=True)
class RealtimeLifecycleConfig:
    """Local lifecycle timeouts without hidden reconnection.

    Parameters
    ----------
    start_timeout_seconds : float or None, default=30.0
        Timeout for runner session creation. ``None`` disables the local timeout.
    close_timeout_seconds : float or None, default=10.0
        Timeout for explicit session close. ``None`` disables the local timeout.
    allow_restart : bool, default=False
        Permit an explicit ``restart`` after close or failure.
    """

    start_timeout_seconds: float | None = 30.0
    close_timeout_seconds: float | None = 10.0
    allow_restart: bool = False

    def __post_init__(self) -> None:
        """Require positive configured timeouts."""
        for name in ("start_timeout_seconds", "close_timeout_seconds"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


class RealtimeSessionProtocol(Protocol):
    """Minimal official Realtime session lifecycle required by the wrapper."""

    async def close(self) -> None:
        """Close the session transport and owned SDK resources."""
        ...


class RealtimeRunnerProtocol(Protocol):
    """Minimal official Realtime runner session factory."""

    def run(self) -> RealtimeSessionProtocol | Awaitable[RealtimeSessionProtocol]:
        """Create one Realtime session."""
        ...


class ManagedRealtimeSession:
    """Explicit lifecycle around an official Realtime runner and session."""

    def __init__(
        self,
        runner: RealtimeRunnerProtocol,
        *,
        lifecycle: RealtimeLifecycleConfig | None = None,
    ) -> None:
        self._runner = runner
        self._lifecycle = lifecycle or RealtimeLifecycleConfig()
        self._session: RealtimeSessionProtocol | None = None
        self._state = RealtimeLifecycleState.CREATED
        self._lock = asyncio.Lock()

    @property
    def raw_runner(self) -> RealtimeRunnerProtocol:
        """Return the underlying official Realtime runner."""
        return self._runner

    @property
    def raw_session(self) -> RealtimeSessionProtocol | None:
        """Return the underlying official session when active or closed."""
        return self._session

    @property
    def state(self) -> RealtimeLifecycleState:
        """Return the current wrapper lifecycle state."""
        return self._state

    @property
    def active(self) -> bool:
        """Return whether session creation completed and close has not begun."""
        return self._state is RealtimeLifecycleState.ACTIVE

    async def start(
        self,
        *,
        operation_context: OperationContext | None = None,
    ) -> RealtimeSessionProtocol:
        """Create one session explicitly and return the raw SDK object."""
        async with self._lock:
            if self._state is RealtimeLifecycleState.ACTIVE:
                assert self._session is not None
                return self._session
            if self._state not in {
                RealtimeLifecycleState.CREATED,
                RealtimeLifecycleState.CLOSED,
                RealtimeLifecycleState.FAILED,
            }:
                raise RuntimeError(
                    f"Cannot start Realtime session from {self._state.value}"
                )
            if (
                self._state is not RealtimeLifecycleState.CREATED
                and not self._lifecycle.allow_restart
            ):
                raise RuntimeError("Realtime session restart is disabled")
            self._state = RealtimeLifecycleState.STARTING

            async def execute() -> RealtimeSessionProtocol:
                try:
                    created = self._runner.run()
                    if inspect.isawaitable(created):
                        session = await cast(
                            Awaitable[RealtimeSessionProtocol],
                            created,
                        )
                    else:
                        session = cast(RealtimeSessionProtocol, created)
                    self._session = session
                    self._state = RealtimeLifecycleState.ACTIVE
                    return session
                except BaseException:
                    self._state = RealtimeLifecycleState.FAILED
                    raise

            return await _with_timeout(
                run_observed_async(operation_context, execute),
                self._lifecycle.start_timeout_seconds,
            )

    async def close(
        self,
        *,
        operation_context: OperationContext | None = None,
    ) -> None:
        """Close the active raw session explicitly and idempotently."""
        async with self._lock:
            if self._state in {
                RealtimeLifecycleState.CREATED,
                RealtimeLifecycleState.CLOSED,
            }:
                self._state = RealtimeLifecycleState.CLOSED
                return
            if self._state is RealtimeLifecycleState.STARTING:
                raise RuntimeError("Cannot close while Realtime session is starting")
            if self._state is RealtimeLifecycleState.CLOSING:
                return
            session = self._session
            if session is None:
                self._state = RealtimeLifecycleState.CLOSED
                return
            self._state = RealtimeLifecycleState.CLOSING

            async def execute() -> None:
                try:
                    result = session.close()
                    if inspect.isawaitable(result):
                        await cast(Awaitable[None], result)
                    self._state = RealtimeLifecycleState.CLOSED
                except BaseException:
                    self._state = RealtimeLifecycleState.FAILED
                    raise

            await _with_timeout(
                run_observed_async(operation_context, execute),
                self._lifecycle.close_timeout_seconds,
            )

    async def restart(
        self,
        *,
        operation_context: OperationContext | None = None,
    ) -> RealtimeSessionProtocol:
        """Explicitly close then create a new session when restart is enabled."""
        if not self._lifecycle.allow_restart:
            raise RuntimeError("Realtime session restart is disabled")
        await self.close(operation_context=operation_context)
        return await self.start(operation_context=operation_context)

    async def __aenter__(self) -> RealtimeSessionProtocol:
        """Start and return the original SDK session."""
        return await self.start()

    async def __aexit__(self, *_: object) -> None:
        """Close the session when leaving the async context."""
        await self.close()


def build_realtime_runner(
    starting_agent: object,
    *,
    config: RealtimeRunnerConfig | None = None,
) -> object:
    """Build the official Realtime runner without creating a session."""
    runner_type = _load_realtime_runner_type()
    kwargs: dict[str, Any] = {"starting_agent": starting_agent}
    sdk_config = (config or RealtimeRunnerConfig()).as_sdk_config()
    if sdk_config:
        kwargs["config"] = sdk_config
    return runner_type(**kwargs)


def manage_realtime_runner(
    runner: RealtimeRunnerProtocol,
    *,
    lifecycle: RealtimeLifecycleConfig | None = None,
) -> ManagedRealtimeSession:
    """Wrap an injected official runner without starting it."""
    return ManagedRealtimeSession(runner, lifecycle=lifecycle)


def _load_realtime_runner_type() -> type[Any]:
    try:
        from agents.realtime import RealtimeRunner
    except ImportError as error:
        raise ImportError(
            "Realtime sessions require a compatible openai-agents installation."
        ) from error
    return RealtimeRunner


async def _with_timeout(awaitable: Awaitable[T], timeout: float | None) -> T:
    if timeout is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=timeout)


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


__all__ = [
    "ManagedRealtimeSession",
    "RealtimeLifecycleConfig",
    "RealtimeLifecycleState",
    "RealtimeRunnerConfig",
    "RealtimeRunnerProtocol",
    "RealtimeSessionConfig",
    "RealtimeSessionProtocol",
    "build_realtime_runner",
    "manage_realtime_runner",
]

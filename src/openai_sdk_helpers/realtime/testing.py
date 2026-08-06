"""Deterministic in-memory Realtime session for tests and examples."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

_SENTINEL = object()


@dataclass(slots=True)
class InMemoryRealtimeSession:
    """Network-free session implementing event iteration and explicit controls.

    Notes
    -----
    This is a deterministic test transport, not a Realtime protocol emulator.
    It records caller inputs and yields only events explicitly pushed by tests.
    """

    messages: list[str] = field(default_factory=list)
    audio_chunks: list[bytes] = field(default_factory=list, repr=False)
    tool_outputs: list[tuple[str, str]] = field(default_factory=list, repr=False)
    interrupt_calls: int = 0
    cancel_calls: int = 0
    close_calls: int = 0
    closed: bool = False
    _queue: asyncio.Queue[object] = field(
        default_factory=asyncio.Queue,
        repr=False,
    )
    _finished: bool = field(default=False, repr=False)

    async def send_message(self, message: str) -> None:
        """Record one non-empty text message."""
        normalized = message.strip()
        if not normalized:
            raise ValueError("message must not be empty")
        if self.closed:
            raise RuntimeError("Realtime test session is closed")
        self.messages.append(normalized)

    async def send_audio(self, audio: bytes) -> None:
        """Record one non-empty copied audio chunk."""
        if not audio:
            raise ValueError("audio must not be empty")
        if self.closed:
            raise RuntimeError("Realtime test session is closed")
        self.audio_chunks.append(bytes(audio))

    async def send_tool_output(self, call_id: str, output: str) -> None:
        """Record one serialized tool output."""
        normalized_call_id = call_id.strip()
        if not normalized_call_id:
            raise ValueError("call_id must not be empty")
        if self.closed:
            raise RuntimeError("Realtime test session is closed")
        self.tool_outputs.append((normalized_call_id, output))

    async def interrupt(self) -> None:
        """Record one explicit interruption request."""
        if self.closed:
            raise RuntimeError("Realtime test session is closed")
        self.interrupt_calls += 1

    async def cancel(self) -> None:
        """Record one explicit response cancellation request."""
        if self.closed:
            raise RuntimeError("Realtime test session is closed")
        self.cancel_calls += 1

    async def close(self) -> None:
        """Close idempotently and finish event iteration."""
        if self.closed:
            return
        self.close_calls += 1
        self.closed = True
        await self.finish()

    async def push_event(self, event: object) -> None:
        """Append one raw event to the deterministic stream."""
        if self._finished:
            raise RuntimeError("Realtime test event stream is finished")
        await self._queue.put(event)

    async def finish(self) -> None:
        """Finish the event stream idempotently."""
        if self._finished:
            return
        self._finished = True
        await self._queue.put(_SENTINEL)

    def __aiter__(self) -> AsyncIterator[object]:
        """Return the deterministic event iterator."""
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[object]:
        while True:
            event = await self._queue.get()
            if event is _SENTINEL:
                break
            yield event


__all__ = ["InMemoryRealtimeSession"]

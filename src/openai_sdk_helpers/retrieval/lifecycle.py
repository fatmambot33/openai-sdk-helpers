"""Typed lifecycle models for OpenAI retrieval resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .contracts import RetrievalOperationResult

ResourceT = TypeVar("ResourceT")


@dataclass(frozen=True, slots=True)
class PollingConfig:
    """Explicit SDK polling and request-timeout configuration.

    Parameters
    ----------
    poll_interval_ms : int, default=1000
        Delay between SDK polling requests in milliseconds.
    timeout_seconds : float or None, default=None
        Timeout forwarded to the official SDK polling helper. ``None`` keeps the
        injected client's timeout configuration.
    terminal_statuses : tuple[str, ...], default=("completed", "failed", "cancelled")
        SDK statuses accepted after the polling helper returns.
    """

    poll_interval_ms: int = 1000
    timeout_seconds: float | None = None
    terminal_statuses: tuple[str, ...] = ("completed", "failed", "cancelled")

    def __post_init__(self) -> None:
        """Validate positive polling values and terminal statuses."""
        if self.poll_interval_ms < 1:
            raise ValueError("poll_interval_ms must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        statuses = tuple(dict.fromkeys(status.strip() for status in self.terminal_statuses))
        if not statuses or any(not status for status in statuses):
            raise ValueError("terminal_statuses must contain non-empty values")
        object.__setattr__(self, "terminal_statuses", statuses)

    def sdk_kwargs(self) -> dict[str, object]:
        """Return keyword arguments accepted by SDK polling helpers."""
        kwargs: dict[str, object] = {"poll_interval_ms": self.poll_interval_ms}
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds
        return kwargs

    def validate_terminal_status(self, status: str | None) -> None:
        """Require a configured terminal status after polling returns.

        Parameters
        ----------
        status : str or None
            SDK-reported vector-store file status.

        Raises
        ------
        RuntimeError
            If the SDK helper returns without a known terminal status.
        """
        if status not in self.terminal_statuses:
            raise RuntimeError(
                "Polling returned a non-terminal vector-store file status: "
                f"{status!r}"
            )


@dataclass(frozen=True, slots=True)
class VectorStoreFileReference:
    """File attachment state inside one OpenAI vector store.

    Parameters
    ----------
    file_id : str
        Underlying Files API identifier.
    vector_store_id : str
        Vector store containing the attachment.
    status : str or None, default=None
        SDK-reported ingestion status.
    last_error : object or None, default=None
        SDK-reported ingestion error without interpretation.
    raw : object or None, default=None
        Underlying official SDK vector-store file resource.
    """

    file_id: str
    vector_store_id: str
    status: str | None = None
    last_error: object | None = None
    raw: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize required identifiers and optional status."""
        file_id = self.file_id.strip()
        vector_store_id = self.vector_store_id.strip()
        if not file_id:
            raise ValueError("file_id must not be empty")
        if not vector_store_id:
            raise ValueError("vector_store_id must not be empty")
        object.__setattr__(self, "file_id", file_id)
        object.__setattr__(self, "vector_store_id", vector_store_id)
        if self.status is not None:
            status = self.status.strip()
            object.__setattr__(self, "status", status or None)


@dataclass(frozen=True, slots=True)
class RetrievalBatchResult(Generic[ResourceT]):
    """Ordered batch outcomes preserving every per-item failure."""

    results: tuple[RetrievalOperationResult[ResourceT], ...]

    def __post_init__(self) -> None:
        """Copy result order into an immutable tuple."""
        object.__setattr__(self, "results", tuple(self.results))

    @property
    def succeeded(self) -> tuple[RetrievalOperationResult[ResourceT], ...]:
        """Return successful outcomes in input order."""
        return tuple(result for result in self.results if result.succeeded)

    @property
    def failed(self) -> tuple[RetrievalOperationResult[ResourceT], ...]:
        """Return failed outcomes in input order."""
        return tuple(result for result in self.results if not result.succeeded)

    @property
    def ok(self) -> bool:
        """Return whether every item succeeded."""
        return not self.failed


__all__ = [
    "PollingConfig",
    "RetrievalBatchResult",
    "VectorStoreFileReference",
]

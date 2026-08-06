"""Tests for retrieval lifecycle observability integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from openai_sdk_helpers import OperationContext, OperationEvent, OperationPhase
from openai_sdk_helpers.retrieval import OpenAIRetrievalClient


class EventCollector:
    """Collect operation events in emission order."""

    def __init__(self) -> None:
        self.events: list[OperationEvent] = []

    def __call__(self, event: OperationEvent) -> None:
        """Store one emitted event."""
        self.events.append(event)


class FilesResource:
    """Minimal SDK-shaped Files resource."""

    def create(self, **kwargs: Any) -> object:
        """Return one synthetic uploaded file resource."""
        return SimpleNamespace(
            id="file_123",
            filename="guide.pdf",
            purpose=kwargs["purpose"],
            bytes=12,
        )


class VectorStoresResource:
    """Minimal SDK-shaped Vector Stores resource."""


class Client:
    """Minimal injected SDK client."""

    def __init__(self) -> None:
        self.files = FilesResource()
        self.vector_stores = VectorStoresResource()


def test_upload_emits_lifecycle_without_exposing_file_content() -> None:
    collector = EventCollector()
    context = OperationContext(
        "retrieval.files.upload",
        metadata={"file_content": "sensitive", "tenant": "example"},
        observers=(collector,),
    )
    client = OpenAIRetrievalClient(Client())

    result = client.upload_file(
        "guide.pdf",
        purpose="user_data",
        operation_context=context,
    )

    assert result.succeeded is True
    assert [event.phase for event in collector.events] == [
        OperationPhase.START,
        OperationPhase.SUCCESS,
    ]
    success = collector.events[-1]
    assert success.result is result
    assert success.diagnostics()["metadata"] == {
        "file_content": "<redacted>",
        "tenant": "example",
    }

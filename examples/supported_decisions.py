"""Credential-free smoke example for the Decisions API helper surface."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from openai_sdk_helpers.decisions import DecisionsClient, decisions_available


class _FakeDecisions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            answers=[SimpleNamespace(type="predicate", probability=0.99)]
        )


resource = _FakeDecisions()
sdk_client = SimpleNamespace(decisions=resource)

assert decisions_available(sdk_client)

client = DecisionsClient(sdk_client)  # type: ignore[arg-type]
answer = client.predicate(
    "The request is safe.",
    instructions="Is this request safe?",
    name="safe",
)

assert answer.type == "predicate"
assert answer.probability == 0.99
assert resource.calls[0]["model"] == "gpt-6-luna"
assert resource.calls[0]["questions"] == (
    {
        "type": "predicate",
        "instructions": "Is this request safe?",
        "name": "safe",
    },
)

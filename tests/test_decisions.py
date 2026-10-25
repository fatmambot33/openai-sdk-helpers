"""Tests for the OpenAI Decisions API helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openai_sdk_helpers.decisions import (
    AsyncDecisionsClient,
    DecisionsClient,
    DecisionsUnavailableError,
    decisions_available,
)
from openai_sdk_helpers.runtime import OperationContext, OperationPhase


class _DecisionResult:
    def __init__(self, answers: list[object]) -> None:
        self.answers = answers


class _SyncDecisions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.answers: list[object] = [
            SimpleNamespace(type="predicate", probability=0.9)
        ]

    def create(self, **kwargs: Any) -> _DecisionResult:
        self.calls.append(kwargs)
        return _DecisionResult(self.answers)


class _AsyncDecisions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.answers: list[object] = [
            SimpleNamespace(type="predicate", probability=0.8)
        ]

    async def create(self, **kwargs: Any) -> _DecisionResult:
        self.calls.append(kwargs)
        return _DecisionResult(self.answers)


def test_capability_detection_and_error_are_actionable() -> None:
    """Detect Decisions structurally and fail before any SDK request."""
    supported = SimpleNamespace(decisions=_SyncDecisions())
    unsupported = SimpleNamespace()

    assert decisions_available(supported)
    assert not decisions_available(unsupported)

    with pytest.raises(DecisionsUnavailableError) as error:
        DecisionsClient(unsupported)  # type: ignore[arg-type]

    assert "openai>=3.37.0" in str(error.value)
    assert error.value.context["feature"] == "decisions"
    assert error.value.context["minimum_openai_version"] == "3.37.0"


def test_generic_create_preserves_resource_result_and_observability() -> None:
    """Forward native request data and preserve the SDK result object."""
    decisions = _SyncDecisions()
    sdk_client = SimpleNamespace(decisions=decisions)
    helper = DecisionsClient(sdk_client)  # type: ignore[arg-type]
    observed = []
    context = OperationContext("decisions.create", observers=(observed.append,))
    questions = (
        {
            "type": "predicate",
            "name": "safe",
            "instructions": "Is this safe?",
        },
    )

    result = helper.create(
        input="A harmless message",
        questions=questions,  # type: ignore[arg-type]
        safety_identifier="user_opaque",
        operation_context=context,
    )

    assert helper.sdk_client is sdk_client
    assert helper.decisions is decisions
    assert result.answers is decisions.answers
    assert decisions.calls == [
        {
            "input": "A harmless message",
            "model": "gpt-6-luna",
            "questions": questions,
            "safety_identifier": "user_opaque",
        }
    ]
    assert [event.phase for event in observed] == [
        OperationPhase.START,
        OperationPhase.SUCCESS,
    ]
    assert observed[-1].result is result


def test_predicate_builds_one_native_question_and_preserves_refusal() -> None:
    """Build a predicate question while returning the native answer union."""
    refusal = SimpleNamespace(type="refusal", name="eligible")
    decisions = _SyncDecisions()
    decisions.answers = [refusal]
    helper = DecisionsClient(  # type: ignore[arg-type]
        SimpleNamespace(decisions=decisions)
    )

    answer = helper.predicate(
        "Candidate profile",
        instructions="Is the candidate eligible?",
        name="eligible",
    )

    assert answer is refusal
    assert decisions.calls[0] == {
        "input": "Candidate profile",
        "model": "gpt-6-luna",
        "questions": (
            {
                "type": "predicate",
                "instructions": "Is the candidate eligible?",
                "name": "eligible",
            },
        ),
    }


def test_choose_builds_official_choice_question() -> None:
    """Forward typed choice values and descriptions unchanged."""
    answer = SimpleNamespace(type="choice", choice="medium")
    decisions = _SyncDecisions()
    decisions.answers = [answer]
    helper = DecisionsClient(  # type: ignore[arg-type]
        SimpleNamespace(decisions=decisions)
    )
    choices = (
        {"value": "low", "description": "Low priority"},
        {"value": "medium", "description": "Medium priority"},
        {"value": "high", "description": "High priority"},
    )

    result = helper.choose(
        "Customer request",
        instructions="Choose the priority.",
        choices=choices,  # type: ignore[arg-type]
        name="priority",
        model="future-decision-model",
    )

    assert result is answer
    assert decisions.calls[0] == {
        "input": "Customer request",
        "model": "future-decision-model",
        "questions": (
            {
                "type": "choice",
                "instructions": "Choose the priority.",
                "choices": choices,
                "name": "priority",
            },
        ),
    }


def test_score_builds_official_ordered_levels() -> None:
    """Forward ordered score levels unchanged."""
    answer = SimpleNamespace(type="score", score=1.6)
    decisions = _SyncDecisions()
    decisions.answers = [answer]
    helper = DecisionsClient(  # type: ignore[arg-type]
        SimpleNamespace(decisions=decisions)
    )
    levels = (
        {"label": "poor", "description": "Does not meet requirements"},
        {"label": "good", "description": "Meets requirements"},
        {"label": "excellent", "description": "Exceeds requirements"},
    )

    result = helper.score(
        "Proposed answer",
        instructions="Score answer quality.",
        levels=levels,  # type: ignore[arg-type]
    )

    assert result is answer
    assert decisions.calls[0]["questions"] == (
        {
            "type": "score",
            "instructions": "Score answer quality.",
            "levels": levels,
        },
    )


@pytest.mark.asyncio
async def test_async_facade_has_parity_and_preserves_native_answer() -> None:
    """Provide async parity without normalizing SDK results."""
    answer = SimpleNamespace(type="choice", choice=True)
    decisions = _AsyncDecisions()
    decisions.answers = [answer]
    sdk_client = SimpleNamespace(decisions=decisions)
    helper = AsyncDecisionsClient(sdk_client)  # type: ignore[arg-type]
    observed = []
    context = OperationContext("decisions.choose", observers=(observed.append,))

    result = await helper.choose(
        "Request",
        instructions="Should this run?",
        choices=({"value": True}, {"value": False}),  # type: ignore[arg-type]
        operation_context=context,
    )

    assert helper.sdk_client is sdk_client
    assert helper.decisions is decisions
    assert result is answer
    assert decisions.calls == [
        {
            "input": "Request",
            "model": "gpt-6-luna",
            "questions": (
                {
                    "type": "choice",
                    "instructions": "Should this run?",
                    "choices": ({"value": True}, {"value": False}),
                },
            ),
        }
    ]
    assert [event.phase for event in observed] == [
        OperationPhase.START,
        OperationPhase.SUCCESS,
    ]


def test_ga_input_json_tool_history_and_maximum_question_batch() -> None:
    """Forward GA JSON/tool history and 400 questions unchanged."""
    resource = _SyncDecisions()
    sdk_client = SimpleNamespace(decisions=resource)
    client = DecisionsClient(sdk_client)  # type: ignore[arg-type]
    evidence = (
        {"role": "system", "content": "Evaluate the supplied evidence."},
        {"role": "user", "content": [{"type": "input_json", "data": {"risk": "low"}}]},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "verified"},
    )
    questions = tuple(
        {"type": "predicate", "name": f"check_{i}", "instructions": "Is it safe?"}
        for i in range(400)
    )

    result = client.create(  # type: ignore[arg-type]
        input=evidence, questions=questions
    )

    assert result.answers is resource.answers
    assert resource.calls[0]["input"] is evidence
    assert resource.calls[0]["questions"] is questions
    assert len(resource.calls[0]["questions"]) == 400
    assert "prompt_cache_key" not in resource.calls[0]
    assert "prompt_cache_options" not in resource.calls[0]


@pytest.mark.parametrize("method", ["predicate", "choose", "score"])
def test_ga_structured_instructions_and_cache_forwarding(method: str) -> None:
    """Forward structured instructions and caching for every shortcut."""
    resource = _SyncDecisions()
    sdk_client = SimpleNamespace(decisions=resource)
    client = DecisionsClient(sdk_client)  # type: ignore[arg-type]
    instructions = (
        {"type": "input_text", "text": "Evaluate the attached policy."},
        {"type": "input_json", "data": {"policy": "approve"}},
    )
    cache_options = {"mode": "explicit", "ttl": "30m"}
    extra: dict[str, Any] = {}
    if method == "choose":
        extra["choices"] = ({"value": "yes"}, {"value": "no"})
    elif method == "score":
        extra["levels"] = ({"label": "low"}, {"label": "high"})

    getattr(client, method)(
        "Test input",
        instructions=instructions,
        prompt_cache_key="account-workflow",
        prompt_cache_options=cache_options,
        **extra,
    )

    assert resource.calls[0]["questions"][0]["instructions"] is instructions
    assert resource.calls[0]["prompt_cache_key"] == "account-workflow"
    assert resource.calls[0]["prompt_cache_options"] is cache_options


@pytest.mark.asyncio
async def test_ga_async_structured_instruction_and_cache_parity() -> None:
    """Preserve GA parameters through asynchronous calls."""
    resource = _AsyncDecisions()
    sdk_client = SimpleNamespace(decisions=resource)
    client = AsyncDecisionsClient(sdk_client)  # type: ignore[arg-type]
    instructions = ({"type": "input_text", "text": "Should this be processed?"},)
    cache_options = {"mode": "implicit", "ttl": "30m"}

    await client.predicate(
        "Message",
        instructions=instructions,
        prompt_cache_key="shared-policy",
        prompt_cache_options=cache_options,
    )
    assert resource.calls[0]["questions"][0]["instructions"] is instructions
    assert resource.calls[0]["prompt_cache_key"] == "shared-policy"
    assert resource.calls[0]["prompt_cache_options"] is cache_options

    questions = ({"type": "predicate", "instructions": "Is this correct?"},)
    messages = (
        {"role": "user", "content": [{"type": "input_json", "data": {"ok": True}}]},
    )
    decision = await client.create(
        input=messages,  # type: ignore[arg-type]
        questions=questions,  # type: ignore[arg-type]
        prompt_cache_options=cache_options,  # type: ignore[arg-type]
    )
    assert decision.answers is resource.answers
    assert resource.calls[1]["input"] is messages
    assert resource.calls[1]["questions"] is questions
    assert resource.calls[1]["prompt_cache_options"] is cache_options

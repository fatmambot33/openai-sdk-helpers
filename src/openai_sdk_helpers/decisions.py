"""Thin helpers for the OpenAI Decisions API."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, cast

from .errors import OpenAISDKError
from .runtime import OperationContext, run_observed_async, run_observed_sync

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openai.types.decision import Answer, Decision
    from openai.types.decision_input_message_param import DecisionInputMessageParam
    from openai.types import decision_create_params

MIN_DECISIONS_OPENAI_VERSION = "3.26.0"
DEFAULT_DECISIONS_MODEL = "gpt-6-luna"


class DecisionsUnavailableError(OpenAISDKError):
    """The supplied OpenAI client does not expose the Decisions API."""


def decisions_available(client: object) -> bool:
    """Return whether a client exposes the Decisions API resource.

    Parameters
    ----------
    client : object
        OpenAI SDK client or compatible test double.

    Returns
    -------
    bool
        ``True`` when ``client.decisions.create`` is available.
    """
    return _find_decisions_resource(client) is not None


class DecisionsClient:
    """Provide synchronous helpers for OpenAI Decisions API requests.

    The facade preserves official SDK question types, result objects,
    exceptions, usage data, and raw resource access. Convenience methods only
    construct one official question and return the corresponding native answer.

    Parameters
    ----------
    sdk_client : OpenAI
        Configured synchronous OpenAI SDK client.

    Raises
    ------
    DecisionsUnavailableError
        If the supplied client does not expose ``decisions.create``.

    Methods
    -------
    create
        Evaluate one or more native Decisions API questions.
    predicate
        Evaluate one predicate question and return its native answer.
    choose
        Evaluate one choice question and return its native answer.
    score
        Evaluate one score question and return its native answer.
    """

    def __init__(self, sdk_client: OpenAI) -> None:
        """Initialize the facade and validate feature availability."""
        self._sdk_client = sdk_client
        self._decisions = _require_decisions_resource(sdk_client)

    @property
    def sdk_client(self) -> OpenAI:
        """Return the underlying configured OpenAI SDK client."""
        return self._sdk_client

    @property
    def decisions(self) -> Any:
        """Return the raw official ``client.decisions`` resource."""
        return self._decisions

    def create(
        self,
        *,
        input: str | Iterable[DecisionInputMessageParam],
        questions: Iterable[decision_create_params.Question],
        model: str = DEFAULT_DECISIONS_MODEL,
        safety_identifier: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Decision:
        """Evaluate one or more Decisions API questions.

        Parameters
        ----------
        input : str or Iterable[DecisionInputMessageParam]
            Shared evidence forwarded unchanged to the official SDK.
        questions : Iterable[decision_create_params.Question]
            Official predicate, choice, or score question definitions.
        model : str, default=DEFAULT_DECISIONS_MODEL
            Decisions model identifier.
        safety_identifier : str or None, default=None
            Optional opaque end-user safety identifier.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        Decision
            Original official SDK Decision object.
        """
        kwargs: dict[str, Any] = {
            "input": input,
            "model": model,
            "questions": questions,
        }
        if safety_identifier is not None:
            kwargs["safety_identifier"] = safety_identifier
        return run_observed_sync(
            operation_context,
            lambda: cast("Decision", self.decisions.create(**kwargs)),
        )

    def predicate(
        self,
        input: str | Iterable[DecisionInputMessageParam],
        *,
        instructions: str,
        name: str | None = None,
        model: str = DEFAULT_DECISIONS_MODEL,
        safety_identifier: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Answer:
        """Evaluate one predicate question.

        Parameters
        ----------
        input : str or Iterable[DecisionInputMessageParam]
            Shared evidence for the decision.
        instructions : str
            Predicate instructions forwarded to the Decisions API.
        name : str or None, default=None
            Optional question name.
        model : str, default=DEFAULT_DECISIONS_MODEL
            Decisions model identifier.
        safety_identifier : str or None, default=None
            Optional opaque end-user safety identifier.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        Answer
            Native predicate answer or refusal returned by the SDK.
        """
        question: dict[str, Any] = {
            "type": "predicate",
            "instructions": instructions,
        }
        if name is not None:
            question["name"] = name
        return self._single_answer(
            input=input,
            question=cast("decision_create_params.Question", question),
            model=model,
            safety_identifier=safety_identifier,
            operation_context=operation_context,
        )

    def choose(
        self,
        input: str | Iterable[DecisionInputMessageParam],
        *,
        instructions: str,
        choices: Iterable[decision_create_params.QuestionQuestionParamChoiceChoice],
        name: str | None = None,
        model: str = DEFAULT_DECISIONS_MODEL,
        safety_identifier: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Answer:
        """Evaluate one fixed-choice question.

        Parameters
        ----------
        input : str or Iterable[DecisionInputMessageParam]
            Shared evidence for the decision.
        instructions : str
            Choice instructions forwarded to the Decisions API.
        choices : Iterable[decision_create_params.QuestionQuestionParamChoiceChoice]
            Official typed choice values and optional descriptions.
        name : str or None, default=None
            Optional question name.
        model : str, default=DEFAULT_DECISIONS_MODEL
            Decisions model identifier.
        safety_identifier : str or None, default=None
            Optional opaque end-user safety identifier.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        Answer
            Native choice answer or refusal returned by the SDK.
        """
        question: dict[str, Any] = {
            "type": "choice",
            "instructions": instructions,
            "choices": choices,
        }
        if name is not None:
            question["name"] = name
        return self._single_answer(
            input=input,
            question=cast("decision_create_params.Question", question),
            model=model,
            safety_identifier=safety_identifier,
            operation_context=operation_context,
        )

    def score(
        self,
        input: str | Iterable[DecisionInputMessageParam],
        *,
        instructions: str,
        levels: Iterable[decision_create_params.QuestionQuestionParamScoreLevel],
        name: str | None = None,
        model: str = DEFAULT_DECISIONS_MODEL,
        safety_identifier: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Answer:
        """Evaluate one ordered-score question.

        Parameters
        ----------
        input : str or Iterable[DecisionInputMessageParam]
            Shared evidence for the decision.
        instructions : str
            Score instructions forwarded to the Decisions API.
        levels : Iterable[decision_create_params.QuestionQuestionParamScoreLevel]
            Official ordered score labels and optional descriptions.
        name : str or None, default=None
            Optional question name.
        model : str, default=DEFAULT_DECISIONS_MODEL
            Decisions model identifier.
        safety_identifier : str or None, default=None
            Optional opaque end-user safety identifier.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        Answer
            Native score answer or refusal returned by the SDK.
        """
        question: dict[str, Any] = {
            "type": "score",
            "instructions": instructions,
            "levels": levels,
        }
        if name is not None:
            question["name"] = name
        return self._single_answer(
            input=input,
            question=cast("decision_create_params.Question", question),
            model=model,
            safety_identifier=safety_identifier,
            operation_context=operation_context,
        )

    def _single_answer(
        self,
        *,
        input: str | Iterable[DecisionInputMessageParam],
        question: decision_create_params.Question,
        model: str,
        safety_identifier: str | None,
        operation_context: OperationContext | None,
    ) -> Answer:
        decision = self.create(
            input=input,
            questions=(question,),
            model=model,
            safety_identifier=safety_identifier,
            operation_context=operation_context,
        )
        return cast("Answer", decision.answers[0])


class AsyncDecisionsClient:
    """Provide asynchronous helpers for OpenAI Decisions API requests.

    Parameters
    ----------
    sdk_client : AsyncOpenAI
        Configured asynchronous OpenAI SDK client.

    Raises
    ------
    DecisionsUnavailableError
        If the supplied client does not expose ``decisions.create``.

    Methods
    -------
    create
        Evaluate one or more native Decisions API questions asynchronously.
    predicate
        Evaluate one predicate question asynchronously.
    choose
        Evaluate one choice question asynchronously.
    score
        Evaluate one score question asynchronously.
    """

    def __init__(self, sdk_client: AsyncOpenAI) -> None:
        """Initialize the facade and validate feature availability."""
        self._sdk_client = sdk_client
        self._decisions = _require_decisions_resource(sdk_client)

    @property
    def sdk_client(self) -> AsyncOpenAI:
        """Return the underlying configured asynchronous OpenAI SDK client."""
        return self._sdk_client

    @property
    def decisions(self) -> Any:
        """Return the raw official asynchronous ``client.decisions`` resource."""
        return self._decisions

    async def create(
        self,
        *,
        input: str | Iterable[DecisionInputMessageParam],
        questions: Iterable[decision_create_params.Question],
        model: str = DEFAULT_DECISIONS_MODEL,
        safety_identifier: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Decision:
        """Evaluate one or more Decisions API questions asynchronously.

        Parameters
        ----------
        input : str or Iterable[DecisionInputMessageParam]
            Shared evidence forwarded unchanged to the official SDK.
        questions : Iterable[decision_create_params.Question]
            Official predicate, choice, or score question definitions.
        model : str, default=DEFAULT_DECISIONS_MODEL
            Decisions model identifier.
        safety_identifier : str or None, default=None
            Optional opaque end-user safety identifier.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        Decision
            Original official SDK Decision object.
        """
        kwargs: dict[str, Any] = {
            "input": input,
            "model": model,
            "questions": questions,
        }
        if safety_identifier is not None:
            kwargs["safety_identifier"] = safety_identifier
        return await run_observed_async(
            operation_context,
            lambda: self.decisions.create(**kwargs),
        )

    async def predicate(
        self,
        input: str | Iterable[DecisionInputMessageParam],
        *,
        instructions: str,
        name: str | None = None,
        model: str = DEFAULT_DECISIONS_MODEL,
        safety_identifier: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Answer:
        """Evaluate one predicate question asynchronously.

        Parameters
        ----------
        input : str or Iterable[DecisionInputMessageParam]
            Shared evidence for the decision.
        instructions : str
            Predicate instructions forwarded to the Decisions API.
        name : str or None, default=None
            Optional question name.
        model : str, default=DEFAULT_DECISIONS_MODEL
            Decisions model identifier.
        safety_identifier : str or None, default=None
            Optional opaque end-user safety identifier.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        Answer
            Native predicate answer or refusal returned by the SDK.
        """
        question: dict[str, Any] = {
            "type": "predicate",
            "instructions": instructions,
        }
        if name is not None:
            question["name"] = name
        return await self._single_answer(
            input=input,
            question=cast("decision_create_params.Question", question),
            model=model,
            safety_identifier=safety_identifier,
            operation_context=operation_context,
        )

    async def choose(
        self,
        input: str | Iterable[DecisionInputMessageParam],
        *,
        instructions: str,
        choices: Iterable[decision_create_params.QuestionQuestionParamChoiceChoice],
        name: str | None = None,
        model: str = DEFAULT_DECISIONS_MODEL,
        safety_identifier: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Answer:
        """Evaluate one fixed-choice question asynchronously.

        Parameters
        ----------
        input : str or Iterable[DecisionInputMessageParam]
            Shared evidence for the decision.
        instructions : str
            Choice instructions forwarded to the Decisions API.
        choices : Iterable[decision_create_params.QuestionQuestionParamChoiceChoice]
            Official typed choice values and optional descriptions.
        name : str or None, default=None
            Optional question name.
        model : str, default=DEFAULT_DECISIONS_MODEL
            Decisions model identifier.
        safety_identifier : str or None, default=None
            Optional opaque end-user safety identifier.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        Answer
            Native choice answer or refusal returned by the SDK.
        """
        question: dict[str, Any] = {
            "type": "choice",
            "instructions": instructions,
            "choices": choices,
        }
        if name is not None:
            question["name"] = name
        return await self._single_answer(
            input=input,
            question=cast("decision_create_params.Question", question),
            model=model,
            safety_identifier=safety_identifier,
            operation_context=operation_context,
        )

    async def score(
        self,
        input: str | Iterable[DecisionInputMessageParam],
        *,
        instructions: str,
        levels: Iterable[decision_create_params.QuestionQuestionParamScoreLevel],
        name: str | None = None,
        model: str = DEFAULT_DECISIONS_MODEL,
        safety_identifier: str | None = None,
        operation_context: OperationContext | None = None,
    ) -> Answer:
        """Evaluate one ordered-score question asynchronously.

        Parameters
        ----------
        input : str or Iterable[DecisionInputMessageParam]
            Shared evidence for the decision.
        instructions : str
            Score instructions forwarded to the Decisions API.
        levels : Iterable[decision_create_params.QuestionQuestionParamScoreLevel]
            Official ordered score labels and optional descriptions.
        name : str or None, default=None
            Optional question name.
        model : str, default=DEFAULT_DECISIONS_MODEL
            Decisions model identifier.
        safety_identifier : str or None, default=None
            Optional opaque end-user safety identifier.
        operation_context : OperationContext or None, default=None
            Optional lifecycle observer context for this request.

        Returns
        -------
        Answer
            Native score answer or refusal returned by the SDK.
        """
        question: dict[str, Any] = {
            "type": "score",
            "instructions": instructions,
            "levels": levels,
        }
        if name is not None:
            question["name"] = name
        return await self._single_answer(
            input=input,
            question=cast("decision_create_params.Question", question),
            model=model,
            safety_identifier=safety_identifier,
            operation_context=operation_context,
        )

    async def _single_answer(
        self,
        *,
        input: str | Iterable[DecisionInputMessageParam],
        question: decision_create_params.Question,
        model: str,
        safety_identifier: str | None,
        operation_context: OperationContext | None,
    ) -> Answer:
        decision = await self.create(
            input=input,
            questions=(question,),
            model=model,
            safety_identifier=safety_identifier,
            operation_context=operation_context,
        )
        return cast("Answer", decision.answers[0])


def _find_decisions_resource(client: object) -> Any | None:
    decisions = getattr(client, "decisions", None)
    create = getattr(decisions, "create", None) if decisions is not None else None
    if not callable(create):
        return None
    return decisions


def _require_decisions_resource(client: object) -> Any:
    decisions = _find_decisions_resource(client)
    if decisions is not None:
        return decisions
    installed_version = _installed_openai_version()
    raise DecisionsUnavailableError(
        "Decisions API helpers require a client exposing client.decisions.create "
        f"(available in openai>={MIN_DECISIONS_OPENAI_VERSION}). Installed openai "
        f"version: {installed_version}. Upgrade openai or use another supported "
        "helper surface.",
        context={
            "feature": "decisions",
            "minimum_openai_version": MIN_DECISIONS_OPENAI_VERSION,
            "installed_openai_version": installed_version,
        },
    )


def _installed_openai_version() -> str:
    try:
        return version("openai")
    except PackageNotFoundError:
        return "not installed"


__all__ = [
    "AsyncDecisionsClient",
    "DEFAULT_DECISIONS_MODEL",
    "DecisionsClient",
    "DecisionsUnavailableError",
    "MIN_DECISIONS_OPENAI_VERSION",
    "decisions_available",
]

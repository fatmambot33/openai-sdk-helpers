"""Agent helper for validating inputs and outputs against guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..structure.validation import ValidationResultStructure
from .base import AgentBase
from .config import AgentConfig
from .prompt_utils import DEFAULT_PROMPT_DIR


class ValidatorAgent(AgentBase):
    """Check user prompts and agent responses against safety guardrails.

    This agent validates inputs and outputs to ensure they comply with safety
    policies and usage guidelines, returning structured validation results with
    recommended actions.

    Examples
    --------
    Validate user input:

    >>> from openai_sdk_helpers.agent import ValidatorAgent
    >>> validator = ValidatorAgent(default_model="gpt-4o-mini")
    >>> result = validator.run_sync("Tell me about Python programming")
    >>> print(result.input_safe)  # True
    >>> print(result.violations)  # []

    Validate both input and output:

    >>> import asyncio
    >>> async def main():
    ...     validator = ValidatorAgent(default_model="gpt-4o-mini")
    ...     result = await validator.run_agent(
    ...         user_input="Summarize this document",
    ...         agent_output="Summary containing PII...",
    ...         policy_notes="No PII in outputs"
    ...     )
    ...     if not result.output_safe:
    ...         print(result.sanitized_output)
    >>> asyncio.run(main())

    Methods
    -------
    run_agent(user_input, agent_output, policy_notes, extra_context)
        Validate user and agent messages and return a structured report.
    """

    def __init__(
        self,
        *,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the validator agent configuration.

        Parameters
        ----------
        prompt_dir : pathlib.Path or None, default=None
            Optional directory containing Jinja prompt templates. Defaults to the
            packaged ``prompt`` directory when not provided.
        default_model : str or None, default=None
            Fallback model identifier when not specified elsewhere.

        Returns
        -------
        None
        """
        config = AgentConfig(
            name="validator",
            description="Validate user input and agent output against guardrails.",
            output_type=ValidationResultStructure,
        )
        prompt_directory = prompt_dir or DEFAULT_PROMPT_DIR
        super().__init__(
            config=config, prompt_dir=prompt_directory, default_model=default_model
        )

    async def run_agent(
        self,
        user_input: str,
        *,
        agent_output: Optional[str] = None,
        policy_notes: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResultStructure:
        """Validate user and agent messages.

        Parameters
        ----------
        user_input : str
            Raw input provided by the user for the agent to evaluate.
        agent_output : str, optional
            Latest agent response to validate against safety guardrails.
            Default ``None`` when only the input should be assessed.
        policy_notes : str, optional
            Additional policy snippets or guardrail expectations to reinforce.
            Default ``None``.
        extra_context : dict, optional
            Additional fields to merge into the validation context. Default ``None``.

        Returns
        -------
        ValidationResultStructure
            Structured validation result describing any violations and actions.
        """
        context: Dict[str, Any] = {"user_input": user_input}
        if agent_output is not None:
            context["agent_output"] = agent_output
        if policy_notes is not None:
            context["policy_notes"] = policy_notes
        if extra_context:
            context.update(extra_context)

        result: ValidationResultStructure = await self.run_async(
            input=user_input,
            context=context,
            output_type=ValidationResultStructure,
        )
        return result


__all__ = ["ValidatorAgent"]

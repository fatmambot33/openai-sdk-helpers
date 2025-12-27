"""Lightweight agents for common text transformations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..structure import SummaryStructure
from .base import BaseAgent, _run_agent
from .config import AgentConfig


class SummarizerAgent(BaseAgent):
    """Generate concise summaries from provided text.

    Methods
    -------
    run_agent(text, metadata)
        Summarize the supplied text with optional metadata context.
    """

    def __init__(
        self,
        *,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
        output_type: type[Any] = SummaryStructure,
    ) -> None:
        """Initialize the summarizer agent configuration.

        Parameters
        ----------
        prompt_dir : pathlib.Path or None, default=None
            Optional directory containing Jinja prompt templates. Defaults to the
            packaged ``prompt`` directory when not provided.
        default_model : str or None, default=None
            Fallback model identifier when not specified elsewhere.
        output_type : type, default=SummaryStructure
            Type describing the expected summary output.

        Returns
        -------
        None
        """
        config = AgentConfig(
            name="summarizer",
            description="Summarize passages into concise findings.",
            output_type=output_type,
        )
        prompt_directory = prompt_dir or _DEFAULT_PROMPT_DIR
        super().__init__(
            config=config, prompt_dir=prompt_directory, default_model=default_model
        )

    async def run_agent(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Return a summary for ``text``.

        Parameters
        ----------
        text : str
            Source content to summarize.
        metadata : dict, optional
            Additional metadata to include in the prompt context. Default ``None``.

        Returns
        -------
        Any
            Structured summary produced by the agent.
        """
        context: Optional[Dict[str, Any]] = None
        if metadata:
            context = {"metadata": metadata}

        result = await _run_agent(
            agent=self.get_agent(),
            agent_input=text,
            agent_context=context,
            output_type=self._output_type,
        )
        return result


class TranslatorAgent(BaseAgent):
    """Translate text into a target language.

    Methods
    -------
    run_agent(text, target_language, context)
        Translate the supplied text into the target language.
    """

    def __init__(
        self,
        *,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the translation agent configuration.

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
            name="translator",
            description="Translate text into the requested language.",
            output_type=str,
        )
        prompt_directory = prompt_dir or _DEFAULT_PROMPT_DIR
        super().__init__(
            config=config, prompt_dir=prompt_directory, default_model=default_model
        )

    async def run_agent(
        self,
        text: str,
        target_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Translate ``text`` to ``target_language``.

        Parameters
        ----------
        text : str
            Source content to translate.
        target_language : str
            Language to translate the content into.
        context : dict, optional
            Additional context values to merge into the prompt. Default ``None``.

        Returns
        -------
        str
            Translated text returned by the agent.
        """
        template_context: Dict[str, Any] = {"target_language": target_language}
        if context:
            template_context.update(context)

        result: str = await _run_agent(
            agent=self.get_agent(),
            agent_input=text,
            agent_context=template_context,
            output_type=str,
        )
        return result


__all__ = ["SummarizerAgent", "TranslatorAgent"]

_DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt"

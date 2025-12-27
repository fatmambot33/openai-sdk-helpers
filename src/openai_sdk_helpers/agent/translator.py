"""Lightweight agent for translating text into a target language."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseAgent, _run_agent
from .config import AgentConfig
from .prompt_utils import DEFAULT_PROMPT_DIR


class TranslatorAgent(BaseAgent):
    """Translate text into a target language.

    Methods
    -------
    run_agent(text, target_language, context)
        Translate the supplied text into the target language.
    run_sync(text, target_language, context)
        Translate the supplied text synchronously.
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
        prompt_directory = prompt_dir or DEFAULT_PROMPT_DIR
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

    def run_sync(
        self,
        text: str,
        target_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Synchronously translate ``text`` to ``target_language``.

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

        result: str = super().run_sync(
            agent_input=text,
            agent_context=template_context,
            output_type=str,
        )
        return result


__all__ = ["TranslatorAgent"]

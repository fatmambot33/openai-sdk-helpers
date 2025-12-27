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
        agent_input: str,
        agent_context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
        *,
        target_language: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Translate ``agent_input`` to ``target_language`` synchronously.

        Parameters
        ----------
        agent_input : str
            Source content to translate.
        agent_context : dict, optional
            Additional context values to merge into the prompt. Default ``None``.
        output_type : type or None, optional
            Optional output type cast for the response. Default ``None``.
        target_language : str, optional
            Target language to translate the content into. Required unless supplied
            within ``agent_context`` or ``kwargs``.
        **kwargs
            Optional keyword arguments. ``context`` is accepted as an alias for
            ``agent_context`` for backward compatibility.

        Returns
        -------
        str
            Translated text returned by the agent.
        """
        merged_context: Dict[str, Any] = {}

        if agent_context:
            merged_context.update(agent_context)
        if "context" in kwargs and kwargs["context"]:
            merged_context.update(kwargs["context"])
        if target_language:
            merged_context["target_language"] = target_language

        if "target_language" not in merged_context:
            msg = "target_language is required for translation"
            raise ValueError(msg)

        result: str = super().run_sync(
            agent_input=agent_input,
            agent_context=merged_context,
            output_type=output_type or str,
        )
        return result


__all__ = ["TranslatorAgent"]

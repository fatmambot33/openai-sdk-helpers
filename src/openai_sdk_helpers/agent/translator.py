"""Lightweight agent for translating text into a target language."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


from ..structure import TranslationStructure
from ..structure.base import StructureBase
from ..environment import DEFAULT_PROMPT_DIR

from .base import AgentBase
from .configuration import AgentConfiguration


class TranslatorAgent(AgentBase):
    """Translate text into a target language.

    This agent provides language translation services using OpenAI models,
    supporting both synchronous and asynchronous execution modes.

    Parameters
    ----------
    prompt_dir : Path or None, default=None
        Optional directory containing Jinja prompt templates. Defaults to the
        packaged ``prompt`` directory when not provided.
    default_model : str or None, default=None
        Fallback model identifier when not specified elsewhere.

    Examples
    --------
    Basic translation:

    >>> from openai_sdk_helpers.agent import TranslatorAgent
    >>> translator = TranslatorAgent(default_model="gpt-4o-mini")
    >>> result = translator.run_sync("Hello world", target_language="Spanish")
    >>> print(result.text)
    'Hola mundo'

    Async translation with context:

    >>> import asyncio
    >>> async def main():
    ...     translator = TranslatorAgent(default_model="gpt-4o-mini")
    ...     result = await translator.run_agent(
    ...         text="Good morning",
    ...         target_language="French",
    ...         context={"formality": "formal"}
    ...     )
    ...     return result
    >>> asyncio.run(main())

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
        prompt_dir : Path or None, default=None
            Optional directory containing Jinja prompt templates. Defaults to the
            packaged ``prompt`` directory when not provided.
        default_model : str or None, default=None
            Fallback model identifier when not specified elsewhere.

        Raises
        ------
        ValueError
            If the default model is not provided.

        Examples
        --------
        >>> translator = TranslatorAgent(default_model="gpt-4o-mini")
        """
        # Use None for template_path unless a file is provided
        configuration = AgentConfiguration(
            name="translator",
            instructions="Agent instructions",
            description="Translate text into the requested language.",
            template_path=None,
            output_structure=TranslationStructure,
            model=default_model,
        )
        super().__init__(
            configuration=configuration,
            default_model=default_model,
        )

    async def run_agent(
        self,
        text: str,
        target_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationStructure:
        """Translate ``text`` to ``target_language``.

        Parameters
        ----------
        text : str
            Source content to translate.
        target_language : str
            Language to translate the content into.
        context : dict or None, default=None
            Additional context values to merge into the prompt.

        Returns
        -------
        TranslationStructure
            Structured translation output from the agent.

        Raises
        ------
        APIError
            If the OpenAI API call fails.

        Examples
        --------
        >>> import asyncio
        >>> async def main():
        ...     result = await translator.run_agent("Hello", "Spanish")
        ...     return result
        >>> asyncio.run(main())
        """
        template_context: Dict[str, Any] = {"target_language": target_language}
        if context:
            template_context.update(context)

        result: TranslationStructure = await self.run_async(
            input=text,
            context=template_context,
        )
        return result

    def run_sync(
        self,
        input: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        output_structure: Optional[type[StructureBase]] = None,
        session: Optional[Any] = None,
        target_language: Optional[str] = None,
    ) -> TranslationStructure:
        """Translate ``input`` to ``target_language`` synchronously.

        Parameters
        ----------
        input : str
            Source content to translate.
        context : dict or None, default=None
            Additional context values to merge into the prompt.
        output_structure : type[StructureBase] or None, default=None
            Optional output type cast for the response.
        target_language : str or None, optional
            Target language to translate the content into. Required unless supplied
            within ``context``.
        session : Session or None, default=None
            Optional session for maintaining conversation history across runs.

        Returns
        -------
        TranslationStructure
            Structured translation output from the agent.

        Raises
        ------
        ValueError
            If ``target_language`` is not provided.

        Examples
        --------
        >>> result = translator.run_sync("Hello", target_language="Spanish")
        """
        merged_context: Dict[str, Any] = {}

        if context:
            merged_context.update(context)
        if target_language:
            merged_context["target_language"] = target_language

        if "target_language" not in merged_context:
            msg = "target_language is required for translation"
            raise ValueError(msg)

        result: TranslationStructure = super().run_sync(
            input=input,
            context=merged_context,
            output_structure=output_structure or self._output_structure,
            session=session,
        )
        return result


__all__ = ["TranslatorAgent"]

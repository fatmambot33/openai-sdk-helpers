"""Lightweight agent for translating text into a target language."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


from ..structure import TranslationStructure
from ..structure.base import StructureBase

from .base import AgentBase
from .configuration import AgentConfiguration


class TranslatorAgent(AgentBase):
    """Translate text into a target language.

    This agent provides language translation services using OpenAI models,
    supporting both synchronous and asynchronous execution modes.

    Parameters
    ----------
    template_path : Path | str | None, default=None
        Optional template file path for prompt rendering.
    model : str | None, default=None
        Model identifier to use for translation.

    Examples
    --------
    Basic translation:

    >>> from openai_sdk_helpers.agent import TranslatorAgent
    >>> translator = TranslatorAgent(model="gpt-4o-mini")
    >>> result = translator.run_sync("Hello world", target_language="Spanish")
    >>> print(result.text)
    'Hola mundo'

    Async translation with context:

    >>> import asyncio
    >>> async def main():
    ...     translator = TranslatorAgent(model="gpt-4o-mini")
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
        template_path: Path | str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the translation agent configuration.

        Parameters
        ----------
        template_path : Path | str | None, default=None
            Optional template file path for prompt rendering.
        model : str | None, default=None
            Model identifier to use for translation.

        Raises
        ------
        ValueError
            If the model is not provided.

        Examples
        --------
        >>> translator = TranslatorAgent(model="gpt-4o-mini")
        """
        configuration = AgentConfiguration(
            name="translator",
            instructions="Agent instructions",
            description="Translate text into the requested language.",
            template_path=template_path,
            output_structure=TranslationStructure,
            model=model,
        )
        super().__init__(configuration=configuration)

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
        input: str | list[dict[str, Any]],
        *,
        context: Optional[Dict[str, Any]] = None,
        output_structure: Optional[type[StructureBase]] = None,
        session: Optional[Any] = None,
        target_language: Optional[str] = None,
    ) -> TranslationStructure:
        """Translate ``input`` to ``target_language`` synchronously.

        Parameters
        ----------
        input : str or list[dict[str, Any]]
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

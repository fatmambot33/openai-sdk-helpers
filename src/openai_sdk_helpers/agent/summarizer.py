"""Lightweight agent for summarizing text."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from ..structure import SummaryStructure
from ..structure.base import StructureBase
from .base import AgentBase
from .configuration import AgentConfiguration
from ..environment import DEFAULT_PROMPT_DIR


class SummarizerAgent(AgentBase):
    """Generate concise summaries from provided text.

    This agent uses OpenAI models to create structured summaries from longer-form
    content. The output follows the ``SummaryStructure`` format by default but
    can be customized with a different output type.

    Parameters
    ----------
    prompt_dir : Path or None, default=None
        Optional directory containing Jinja prompt templates. Defaults to the
        packaged ``prompt`` directory when not provided.
    default_model : str or None, default=None
        Fallback model identifier when not specified elsewhere.
    output_structure : type[StructureBase], default=SummaryStructure
        Type describing the expected summary output.

    Examples
    --------
    Basic usage with default settings:

    >>> from openai_sdk_helpers.agent import SummarizerAgent
    >>> summarizer = SummarizerAgent(default_model="gpt-4o-mini")
    >>> summary = summarizer.run_sync("Long text to summarize...")
    >>> print(summary.text)

    With custom metadata:

    >>> import asyncio
    >>> async def main():
    ...     summarizer = SummarizerAgent(default_model="gpt-4o-mini")
    ...     result = await summarizer.run_agent(
    ...         text="Article content...",
    ...         metadata={"source": "news.txt", "date": "2025-01-01"}
    ...     )
    ...     return result
    >>> asyncio.run(main())

    Methods
    -------
    run_agent(text, metadata)
        Summarize the supplied text with optional metadata context.
    """

    def __init__(
        self,
        *,
        template_path: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the summarizer agent configuration.

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
        >>> summarizer = SummarizerAgent(default_model="gpt-4o-mini")
        """
        configuration = AgentConfiguration(
            name="summarizer",
            instructions="Agent instructions",
            description="Summarize passages into concise findings.",
            template_path=template_path,
            output_structure=SummaryStructure,
            model=default_model,
        )

        super().__init__(
            configuration=configuration,
            default_model=default_model,
        )

    async def run_agent(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Generate a summary for ``text``.

        Parameters
        ----------
        text : str
            Source content to summarize.
        metadata : dict or None, default=None
            Additional metadata to include in the prompt context.

        Returns
        -------
        Any
            Structured summary produced by the agent.

        Raises
        ------
        APIError
            If the OpenAI API call fails.
        """
        context: Optional[Dict[str, Any]] = None
        if metadata:
            context = {"metadata": metadata}

        result = await self.run_async(
            input=text,
            context=context,
            output_structure=self._output_structure,
        )
        return result


__all__ = ["SummarizerAgent"]

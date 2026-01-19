"""Lightweight agent for summarizing text."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..structure import SummaryStructure
from ..structure.base import StructureBase
from .base import AgentBase
from .configuration import AgentConfiguration


class SummarizerAgent(AgentBase):
    """Generate concise summaries from provided text.

    This agent uses OpenAI models to create structured summaries from longer-form
    content. The output follows the ``SummaryStructure`` format by default but
    can be customized with a different output type.

    Parameters
    ----------
    template_path : Path | str | None, default=None
        Optional template file path for prompt rendering.
    model : str | None, default=None
        Model identifier to use for summarization.
    output_structure : type[StructureBase], default=SummaryStructure
        Type describing the expected summary output.

    Examples
    --------
    Basic usage with default settings:

    >>> from openai_sdk_helpers.agent import SummarizerAgent
    >>> summarizer = SummarizerAgent(model="gpt-4o-mini")
    >>> summary = summarizer.run_sync("Long text to summarize...")
    >>> print(summary.text)

    With custom metadata:

    >>> import asyncio
    >>> async def main():
    ...     summarizer = SummarizerAgent(model="gpt-4o-mini")
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
        template_path: Path | str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the summarizer agent configuration.

        Parameters
        ----------
        template_path : Path | str | None, default=None
            Optional template file path for prompt rendering.
        model : str | None, default=None
            Model identifier to use for summarization.

        Raises
        ------
        ValueError
            If the model is not provided.

        Examples
        --------
        >>> summarizer = SummarizerAgent(model="gpt-4o-mini")
        """
        configuration = AgentConfiguration(
            name="summarizer",
            instructions="Agent instructions",
            description="Summarize passages into concise findings.",
            template_path=template_path,
            output_structure=SummaryStructure,
            model=model,
        )

        super().__init__(configuration=configuration)

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

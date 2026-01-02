"""Lightweight agent for summarizing text."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..structure import SummaryStructure
from .base import AgentBase
from .config import AgentConfig
from .prompt_utils import DEFAULT_PROMPT_DIR


class SummarizerAgent(AgentBase):
    """Generate concise summaries from provided text.

    This agent uses OpenAI models to create structured summaries from longer-form
    content. The output follows the ``SummaryStructure`` format by default but
    can be customized with a different output type.

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
        prompt_directory = prompt_dir or DEFAULT_PROMPT_DIR
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

        result = await self.run_async(
            input=text,
            context=context,
            output_type=self._output_type,
        )
        return result


__all__ = ["SummarizerAgent"]

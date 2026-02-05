"""LangExtract-backed agent for structured document extraction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from ..extract import DocumentExtractor
from ..prompt import PromptRenderer
from ..structure.extraction import (
    AnnotatedDocumentStructure,
    DocumentExtractorConfig,
    DocumentStructure,
    ExampleDataStructure,
)


class ExtractorAgent:
    """Extract structured data from documents using LangExtract.

    Parameters
    ----------
    prompt_description : str | None
        Prompt description used by LangExtract. Required when template_path
        is not provided.
    template_path : Path | str | None, default=None
        Optional Jinja template path for rendering the prompt description.
    template_context : dict[str, object] | None, default=None
        Optional context values for rendering template_path.
    include_env : bool, default=True
        Whether to include environment variables in the template context.
    examples : Sequence[ExampleDataStructure]
        Example payloads supplied to LangExtract.
    model : str
        Model identifier to pass to LangExtract.
    max_workers : int, default=1
        Maximum number of workers for concurrent extraction.

    Methods
    -------
    extract(documents)
        Extract structured data from document structures.
    extract_text(text, additional_context=None)
        Extract structured data from raw text input.
    from_config(config, model, max_workers=1)
        Build an extractor agent from a configuration object.
    """

    def __init__(
        self,
        prompt_description: str | None,
        examples: Sequence[ExampleDataStructure],
        model: str,
        *,
        template_path: Path | str | None = None,
        template_context: dict[str, object] | None = None,
        include_env: bool = True,
        max_workers: int = 1,
    ) -> None:
        """Initialize the extractor agent.

        Parameters
        ----------
        prompt_description : str or None
            Prompt description used by LangExtract. Required when template_path
            is not provided.
        examples : Sequence[ExampleDataStructure]
            Example payloads supplied to LangExtract.
        model : str
            Model identifier to pass to LangExtract.
        template_path : Path | str | None, default=None
            Optional Jinja template path for rendering the prompt description.
        template_context : dict[str, object] or None, default=None
            Optional context values for rendering template_path.
        include_env : bool, default=True
            Whether to include environment variables in the template context.
        max_workers : int, default=1
            Maximum number of workers for concurrent extraction.

        Raises
        ------
        ValueError
            If no examples are provided.

        Examples
        --------
        >>> agent = ExtractorAgent(
        ...     prompt_description="Extract entities.",
        ...     examples=[ExampleDataStructure(text="Hello")],
        ...     model="gpt-4o-mini",
        ... )
        """
        if template_path is not None:
            prompt_description = self._render_template(
                template_path,
                context=template_context,
                include_env=include_env,
            )
        if not prompt_description:
            raise ValueError(
                "prompt_description is required when template_path is None."
            )
        self._extractor = DocumentExtractor(
            prompt_description=prompt_description,
            examples=examples,
            model_id=model,
            max_workers=max_workers,
        )

    @staticmethod
    def _render_template(
        template_path: Path | str,
        *,
        context: dict[str, object] | None = None,
        include_env: bool = True,
    ) -> str:
        """Render a prompt template for the extractor.

        Parameters
        ----------
        template_path : Path | str
            Path to the Jinja template to render.
        context : dict[str, object] or None, default=None
            Context values supplied to the template.
        include_env : bool, default=True
            Whether to include environment variables in the template context.

        Returns
        -------
        str
            Rendered prompt description.
        """
        renderer = PromptRenderer()
        rendered_context: dict[str, Any] = {}
        if include_env:
            rendered_context["env"] = dict(os.environ)
        if context:
            rendered_context.update(context)
        return renderer.render(str(template_path), context=rendered_context)

    @classmethod
    def from_config(
        cls,
        config: DocumentExtractorConfig,
        *,
        model: str,
        max_workers: int = 1,
    ) -> "ExtractorAgent":
        """Build an extractor agent from a configuration object.

        Parameters
        ----------
        config : DocumentExtractorConfig
            Configuration describing the extraction prompt and examples.
        model : str
            Model identifier to pass to LangExtract.
        max_workers : int, default=1
            Maximum number of workers for concurrent extraction.

        Returns
        -------
        ExtractorAgent
            Configured extractor agent instance.

        Examples
        --------
        >>> config = DocumentExtractorConfig(
        ...     name="example",
        ...     prompt_description="Extract entities.",
        ...     extraction_classes=["entity"],
        ...     examples=[ExampleDataStructure(text="Example")],
        ... )
        >>> agent = ExtractorAgent.from_config(config, model="gpt-4o-mini")
        """
        return cls(
            prompt_description=config.prompt_description,
            examples=config.examples,
            model=model,
            max_workers=max_workers,
        )

    def extract(
        self,
        documents: DocumentStructure | list[DocumentStructure],
    ) -> list[AnnotatedDocumentStructure]:
        """Extract structured data from document structures.

        Parameters
        ----------
        documents : DocumentStructure or list[DocumentStructure]
            Document structures to extract data from.

        Returns
        -------
        list[AnnotatedDocumentStructure]
            Extraction results for the provided documents.

        Examples
        --------
        >>> agent = ExtractorAgent(
        ...     prompt_description="Extract entities.",
        ...     examples=[ExampleDataStructure(text="Example")],
        ...     model="gpt-4o-mini",
        ... )
        >>> document = DocumentStructure(text="Hello world")
        >>> agent.extract(document)
        [AnnotatedDocumentStructure(...)]
        """
        return self._extractor.extract(documents)

    def extract_text(
        self,
        text: str | Sequence[str],
        *,
        additional_context: str | None = None,
    ) -> list[AnnotatedDocumentStructure]:
        """Extract structured data from raw text input.

        Parameters
        ----------
        text : str or Sequence[str]
            Raw text content to extract from.
        additional_context : str or None, default=None
            Optional additional context to attach to each document.

        Returns
        -------
        list[AnnotatedDocumentStructure]
            Extraction results for the provided text.

        Examples
        --------
        >>> agent = ExtractorAgent(
        ...     prompt_description="Extract entities.",
        ...     examples=[ExampleDataStructure(text="Example")],
        ...     model="gpt-4o-mini",
        ... )
        >>> agent.extract_text("Hello world")
        [AnnotatedDocumentStructure(...)]
        """
        if isinstance(text, str):
            documents = [
                DocumentStructure(text=text, additional_context=additional_context)
            ]
        else:
            documents = [
                DocumentStructure(text=item, additional_context=additional_context)
                for item in text
            ]
        return self.extract(documents)


__all__ = ["ExtractorAgent"]

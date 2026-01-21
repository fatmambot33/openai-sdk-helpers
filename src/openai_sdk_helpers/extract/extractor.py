"""Document extraction helpers powered by LangExtract."""

from __future__ import annotations

import dataclasses
import os
import typing
import langextract as lx

from ..errors import ExtractionError
from ..structure.extraction import AnnotatedDocument, Document


class DocumentExtractor:
    """Extract structured data from documents using LangExtract.

    Parameters
    ----------
    model : str
        Model identifier to pass to LangExtract.
    schema : SchemaLike | None, optional
        Schema dictionary for building prompts and examples. Default is None.
    examples : Any | None, optional
        LangExtract examples payload. Default is None.
    prompt : str | None, optional
        Prompt string to override schema-derived prompts. Default is None.
    **defaults
        Default keyword arguments forwarded to LangExtract.

    Methods
    -------
    extract(text_or_docs, **kwargs)
        Extract structured data from one or more documents.
    aextract(text_or_docs, **kwargs)
        Asynchronously extract structured data using a thread.
    """

    def __init__(
        self,
        prompt_description: str,
        examples: typing.Sequence[typing.Any],
        model_id: str,
        max_workers: int = 1,
    ) -> None:
        """Initialize the extractor."""
        self.model_id = model_id
        self.prompt = prompt_description
        self.examples = examples
        self.max_workers = max_workers

    def extract(self, input_text: Document | list[Document]) -> list[AnnotatedDocument]:
        """Run the extraction."""
        if isinstance(input_text, Document):
            input_text = [input_text]
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=self.prompt,
            examples=self.examples,
            model_id=self.model_id,  # Automatically selects OpenAI provider
            api_key=os.environ.get("OPENAI_API_KEY"),
            fence_output=True,
            use_schema_constraints=False,
        )
        if isinstance(result, list):
            extracted_items = []
            for doc in result:
                extraction = self._convert_extraction(doc)
                extracted_items.append(extraction)
            return extracted_items

        return [self._convert_extraction(result)]

    @staticmethod
    def _convert_extraction(data: typing.Any) -> AnnotatedDocument:
        """Convert a LangExtract payload into an AnnotatedDocument."""
        if dataclasses.is_dataclass(data):
            return AnnotatedDocument.from_dataclass(data)
        return AnnotatedDocument.model_validate(data)


__all__ = ["DocumentExtractor", "ExtractionError"]

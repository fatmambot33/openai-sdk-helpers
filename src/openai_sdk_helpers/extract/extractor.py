"""Document extraction helpers powered by LangExtract."""

from __future__ import annotations

import os
import typing
import langextract as lx
from langextract.core.data import AnnotatedDocument as LXAnnotatedDocument

from ..errors import ExtractionError
from ..structure.extraction import (
    AnnotatedDocumentStructure,
    DocumentStructure,
    ExampleDataStructure,
)


class DocumentExtractor:
    """Extract structured data from documents using LangExtract.

    Parameters
    ----------
    prompt_description : str
        Prompt description used by LangExtract.
    examples : Sequence[ExampleDataStructure]
        Example payloads supplied to LangExtract.
    model_id : str
        Model identifier to pass to LangExtract.
    max_workers : int, optional
        Maximum number of workers for concurrent extraction. Default is 1.

    Methods
    -------
    extract(input_text)
        Extract structured data from one or more documents.
    """

    def __init__(
        self,
        prompt_description: str,
        examples: typing.Sequence[ExampleDataStructure],
        model_id: str,
        max_workers: int = 1,
    ) -> None:
        """Initialize the extractor.

        Parameters
        ----------
        prompt_description : str
            Prompt description used by LangExtract.
        examples : Sequence[ExampleDataStructure]
            Example payloads supplied to LangExtract.
        model_id : str
            Model identifier to pass to LangExtract.
        max_workers : int, optional
            Maximum number of workers for concurrent extraction. Default is 1.
        """
        if not examples:
            raise ValueError(
                "Examples are required for reliable extraction. "
                "Provide at least one ExampleDataStructure instance."
            )
        self.model_id = model_id
        self.prompt = prompt_description
        self.examples = examples
        self.max_workers = max_workers

    def extract(
        self, input_text: DocumentStructure | list[DocumentStructure]
    ) -> list[AnnotatedDocumentStructure]:
        """Run the extraction.

        Parameters
        ----------
        input_text : DocumentStructure | list[DocumentStructure]
            Document or list of documents to extract data from.

        Returns
        -------
        list[AnnotatedDocumentStructure]
            Extracted items for the provided documents.
        """
        if isinstance(input_text, DocumentStructure):
            input_documents = [input_text]
        else:
            input_documents = input_text
        documents = DocumentStructure.to_dataclass_list(input_documents)
        examples = ExampleDataStructure.to_dataclass_list(self.examples)
        result = lx.extract(
            text_or_documents=documents,
            prompt_description=self.prompt,
            examples=examples,
            model_id=self.model_id,  # Automatically selects OpenAI provider
            api_key=os.environ.get("OPENAI_API_KEY"),
            fence_output=True,
            use_schema_constraints=False,
        )

        def _convert(data: typing.Any) -> AnnotatedDocumentStructure:
            if isinstance(data, LXAnnotatedDocument):
                return AnnotatedDocumentStructure.from_dataclass(data)
            return AnnotatedDocumentStructure.model_validate(data)

        if isinstance(result, list):
            return [_convert(doc) for doc in result]

        return [_convert(result)]


__all__ = ["DocumentExtractor", "ExtractionError"]

"""Document extraction helpers powered by LangExtract."""

from __future__ import annotations

import os
import typing
import langextract as lx
from langextract.core.data import AnnotatedDocument as LXAnnotatedDocument

from ..errors import ExtractionError
from ..structure.extraction import AnnotatedDocument, Document, ExampleData


class DocumentExtractor:
    """Extract structured data from documents using LangExtract.

    Parameters
    ----------
    prompt_description : str
        Prompt description used by LangExtract.
    examples : Sequence[Any]
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
        examples: typing.Sequence[ExampleData],
        model_id: str,
        max_workers: int = 1,
    ) -> None:
        """Initialize the extractor.

        Parameters
        ----------
        prompt_description : str
            Prompt description used by LangExtract.
        examples : Sequence[ExampleData]
            Example payloads supplied to LangExtract.
        model_id : str
            Model identifier to pass to LangExtract.
        max_workers : int, optional
            Maximum number of workers for concurrent extraction. Default is 1.
        """
        if not examples:
            raise ValueError(
                "Examples are required for reliable extraction. "
                "Provide at least one ExampleData instance."
            )
        self.model_id = model_id
        self.prompt = prompt_description
        self.examples = examples
        self.max_workers = max_workers

    def extract(self, input_text: Document | list[Document]) -> list[AnnotatedDocument]:
        """Run the extraction.

        Parameters
        ----------
        input_text : Document | list[Document]
            Document or list of documents to extract data from.

        Returns
        -------
        list[AnnotatedDocument]
            Extracted items for the provided documents.
        """
        if isinstance(input_text, Document):
            input_text = [input_text]
        examples = [
            example.to_dataclass() if isinstance(example, ExampleData) else example
            for example in self.examples
        ]
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=self.prompt,
            examples=examples,
            model_id=self.model_id,  # Automatically selects OpenAI provider
            api_key=os.environ.get("OPENAI_API_KEY"),
            fence_output=True,
            use_schema_constraints=False,
        )

        def _convert(data: typing.Any) -> AnnotatedDocument:
            if isinstance(data, LXAnnotatedDocument):
                return AnnotatedDocument.from_dataclass(data)
            return AnnotatedDocument.model_validate(data)

        if isinstance(result, list):
            return [_convert(doc) for doc in result]

        return [_convert(result)]


__all__ = ["DocumentExtractor", "ExtractionError"]

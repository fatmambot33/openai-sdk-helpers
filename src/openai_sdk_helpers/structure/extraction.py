"""Structured extraction result models."""

from __future__ import annotations

from typing import Any
import uuid
from enum import Enum, IntEnum
from langextract.core.data import (
    AlignmentStatus as LXAlignmentStatus,
    AnnotatedDocument as LXAnnotatedDocument,
    CharInterval as LXCharInterval,
    Document as LXDocument,
    ExampleData as LXExampleData,
    Extraction as LXExtraction,
)

from langextract.core import tokenizer as LXtokenizer
from .base import StructureBase, spec_field


class CharInterval(StructureBase):
    """Class for representing a character interval.

    Attributes
    ----------
      start_pos: The starting position of the interval (inclusive).
      end_pos: The ending position of the interval (exclusive).

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``CharInterval`` dataclass.
    """

    start_pos: int = spec_field(
        "start_pos",
        description="The starting position of the interval (inclusive).",
        default=0,
    )
    end_pos: int = spec_field(
        "end_pos",
        description="The ending position of the interval (exclusive).",
        default=0,
    )

    def to_dataclass(self) -> LXCharInterval:
        """Convert to LangExtract CharInterval dataclass."""
        return LXCharInterval(
            start_pos=self.start_pos,
            end_pos=self.end_pos,
        )


class AlignmentStatus(Enum):
    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"


class TokenInterval(StructureBase):
    """Represents an interval over tokens in tokenized text.

    The interval is defined by a start index (inclusive) and an end index
    (exclusive).

    Attributes
    ----------
      start_index: The index of the first token in the interval.
      end_index: The index one past the last token in the interval.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``TokenInterval`` dataclass.
    """

    start_index: int = spec_field(
        "start_index",
        description="The index of the first token in the interval.",
        default=0,
    )
    end_index: int = spec_field(
        "end_index",
        description="The index one past the last token in the interval.",
        default=0,
    )

    def to_dataclass(self) -> LXtokenizer.TokenInterval:
        """Convert to LangExtract TokenInterval dataclass."""
        return LXtokenizer.TokenInterval(
            start_index=self.start_index,
            end_index=self.end_index,
        )


class TokenType(IntEnum):
    """Enumeration of token types produced during tokenization.

    Attributes
    ----------
      WORD: Represents an alphabetical word token.
      NUMBER: Represents a numeric token.
      PUNCTUATION: Represents punctuation characters.
    """

    WORD = 0
    NUMBER = 1
    PUNCTUATION = 2


class Token(StructureBase):
    """Represents a token extracted from text.

    Each token is assigned an index and classified into a type (word, number,
    punctuation, or acronym). The token also records the range of characters
    (its CharInterval) that correspond to the substring from the original text.
    Additionally, it tracks whether it follows a newline.

    Attributes
    ----------
      index: The position of the token in the sequence of tokens.
      token_type: The type of the token, as defined by TokenType.
      char_interval: The character interval within the original text that this
        token spans.
      first_token_after_newline: True if the token immediately follows a newline
        or carriage return.
    """

    index: int = spec_field(
        "index",
        description="The position of the token in the sequence of tokens.",
    )
    token_type: TokenType = spec_field(
        "token_type",
        description="The type of the token, as defined by TokenType.",
    )
    char_interval: CharInterval = spec_field(
        "char_interval",
        description="The character interval within the original text that this token spans.",
        default_factory=CharInterval,
    )
    first_token_after_newline: bool = spec_field(
        "first_token_after_newline",
        description="True if the token immediately follows a newline or carriage return.",
        default=False,
    )

    def to_dataclass(self) -> LXtokenizer.Token:
        """Convert to LangExtract Token dataclass."""
        return LXtokenizer.Token(
            index=self.index,
            token_type=LXtokenizer.TokenType(self.token_type),
            char_interval=self.char_interval.to_dataclass(),
            first_token_after_newline=self.first_token_after_newline,
        )


class TokenizedText(StructureBase):
    """Holds the result of tokenizing a text string.

    Attributes
    ----------
      text: The text that was tokenized. For UnicodeTokenizer, this is
        NOT normalized to NFC (to preserve indices).
      tokens: A list of Token objects extracted from the text.
    """

    text: str = spec_field(
        "text",
        description="The text that was tokenized.",
        allow_null=False,
    )
    tokens: list[Token] = spec_field(
        "tokens",
        description="A list of Token objects extracted from the text.",
        allow_null=True,
        default_factory=list,
    )

    def to_dataclass(self) -> LXtokenizer.TokenizedText:
        """Convert to LangExtract TokenizedText dataclass."""
        lx_tokens = [token.to_dataclass() for token in self.tokens]
        return LXtokenizer.TokenizedText(
            text=self.text,
            tokens=lx_tokens,
        )


class Extraction(StructureBase):
    """Represent a single extraction from a document.

    Attributes
    ----------
    extraction_class : str
        Label or class assigned to the extracted item.
    extraction_text : str
        Raw text captured for the extracted item.
    description : str | None
        Optional description of the extracted item.
    attributes : dict[str, Any]
        Additional attributes attached to the item. Default is an empty dict.
    char_interval : CharInterval | None
        Character interval in the source text.
    alignment_status : AlignmentStatus | None
        Alignment status of the extracted item.
    extraction_index : int | None
        Index of the extraction in the list of extractions.
    group_index : int | None
        Index of the group this item belongs to, if applicable.
    token_interval : TokenInterval | None
        Token interval of the extracted item.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``Extraction`` dataclass.
    from_dataclass(data)
        Create an extraction from a LangExtract dataclass.
    """

    extraction_class: str = spec_field(
        "extraction_class",
        allow_null=False,
        description="Label or class for the extracted item.",
    )
    extraction_text: str = spec_field(
        "extraction_text",
        allow_null=False,
        description="Raw text captured for the extracted item.",
    )
    description: str = spec_field(
        "description",
        allow_null=True,
        description="Optional description of the extracted item.",
    )
    attributes: dict[str, Any] | None = spec_field(
        "attributes",
        default=None,
        description="Additional attributes attached to the item.",
    )
    char_interval: CharInterval = spec_field(
        "char_interval",
        allow_null=True,
        description="Character interval of the extracted item in the source text.",
    )
    alignment_status: AlignmentStatus = spec_field(
        "alignment_status",
        allow_null=True,
        description="Alignment status of the extracted item.",
    )
    extraction_index: int = spec_field(
        "extraction_index",
        description="Index of the extraction in the list of extractions.",
        allow_null=True,
    )
    group_index: int = spec_field(
        "group_index",
        description="Index of the group this item belongs to, if applicable.",
        allow_null=True,
    )

    token_interval: TokenInterval | None = spec_field(
        "token_interval",
        description="Token interval of the extracted item.",
        allow_null=True,
    )

    def to_dataclass(self) -> LXExtraction:
        """Convert to LangExtract Extraction dataclass.

        Returns
        -------
        LXExtraction
            LangExtract extraction dataclass instance.
        """
        char_interval = (
            self.char_interval.to_dataclass() if self.char_interval is not None else None
        )
        alignment_status = None
        if self.alignment_status is not None:
            alignment_status = LXAlignmentStatus(self.alignment_status.value)
        token_interval = (
            self.token_interval.to_dataclass()
            if self.token_interval is not None
            else None
        )
        return LXExtraction(
            extraction_class=self.extraction_class,
            extraction_text=self.extraction_text,
            char_interval=char_interval,
            alignment_status=alignment_status,
            extraction_index=self.extraction_index,
            group_index=self.group_index,
            description=self.description,
            attributes=self.attributes,
            token_interval=token_interval,
        )

    @classmethod
    def from_dataclass(cls, data: LXExtraction) -> "Extraction":
        """Create an extraction from a LangExtract dataclass.

        Parameters
        ----------
        data : LXExtraction
            LangExtract extraction dataclass instance.

        Returns
        -------
        Extraction
            Structured extraction model.
        """
        if not isinstance(data, LXExtraction):
            return super().from_dataclass(data)
        char_interval = (
            CharInterval.from_dataclass(data.char_interval)
            if data.char_interval is not None
            else None
        )
        alignment_status = (
            AlignmentStatus(data.alignment_status.value)
            if data.alignment_status is not None
            else None
        )
        token_interval = (
            TokenInterval.from_dataclass(data.token_interval)
            if data.token_interval is not None
            else None
        )
        return cls(
            extraction_class=data.extraction_class,
            extraction_text=data.extraction_text,
            char_interval=char_interval,
            alignment_status=alignment_status,
            extraction_index=data.extraction_index,
            group_index=data.group_index,
            description=data.description,
            attributes=data.attributes,
            token_interval=token_interval,
        )


class ExampleData(StructureBase):
    """Represent example data for structured prompting.

    Attributes
    ----------
    text : str
        Raw text for the example.
    extractions : list[Extraction]
        Extractions associated with the text. Default is an empty list.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``ExampleData`` dataclass.
    """

    text: str = spec_field(
        "text",
        allow_null=False,
        description="Raw text for the example.",
    )
    extractions: list[Extraction] = spec_field(
        "extractions",
        description="Extractions associated with the text.",
        default_factory=list,
    )

    def to_dataclass(self) -> LXExampleData:
        """Convert to LangExtract ExampleData dataclass.

        Returns
        -------
        LXExampleData
            LangExtract example dataclass instance.
        """
        lx_extractions = [extraction.to_dataclass() for extraction in self.extractions]
        return LXExampleData(
            text=self.text,
            extractions=lx_extractions,
        )

    @classmethod
    def from_dataclass(cls, data: LXExampleData) -> "ExampleData":
        """Create example data from a LangExtract dataclass.

        Parameters
        ----------
        data : LXExampleData
            LangExtract example dataclass instance.

        Returns
        -------
        ExampleData
            Structured example data model.
        """
        if not isinstance(data, LXExampleData):
            return super().from_dataclass(data)
        extractions = (
            [Extraction.from_dataclass(item) for item in data.extractions]
            if data.extractions is not None
            else []
        )
        return cls(text=data.text, extractions=extractions)


class AnnotatedDocument(StructureBase):
    """Represent a document annotated with extractions.

    Attributes
    ----------
    document_id : str | None
        Identifier for the document.
    extractions : list[Extraction] | None
        Extractions associated with the document.
    text : str | None
        Raw text representation of the document.
    tokenized_text : TokenizedText | None
        Tokenized text for the document.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``AnnotatedDocument`` dataclass.
    from_dataclass(data)
        Create an annotated document from a LangExtract dataclass.
    """

    document_id: str | None = spec_field(
        "document_id",
        description="Identifier for the document.",
        allow_null=True,
    )
    extractions: list[Extraction] | None = spec_field(
        "extractions",
        description="Extractions associated with the document.",
        allow_null=True,
        default_factory=list,
    )
    text: str | None = spec_field(
        "text",
        description="Raw text representation of the document.",
        allow_null=True,
    )
    tokenized_text: TokenizedText | None = spec_field(
        "tokenized_text",
        description="Tokenized representation of the document text.",
        allow_null=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """Populate default identifiers and tokenized text after validation."""
        if self.document_id is None:
            self.document_id = f"doc_{uuid.uuid4().hex[:8]}"
        if self.text and self.tokenized_text is None:
            tokenized = LXtokenizer.tokenize(self.text)
            self.tokenized_text = TokenizedText.from_dataclass(tokenized)

    def to_dataclass(self) -> LXAnnotatedDocument:
        """Convert to LangExtract AnnotatedDocument dataclass.

        Returns
        -------
        LXAnnotatedDocument
            LangExtract annotated document dataclass instance.
        """
        lx_extractions = (
            [extraction.to_dataclass() for extraction in self.extractions]
            if self.extractions is not None
            else None
        )
        lx_doc = LXAnnotatedDocument(
            document_id=self.document_id,
            extractions=lx_extractions,
            text=self.text,
        )
        if self.tokenized_text is not None:
            lx_doc.tokenized_text = self.tokenized_text.to_dataclass()
        return lx_doc

    @classmethod
    def from_dataclass(cls, data: LXAnnotatedDocument) -> "AnnotatedDocument":
        """Create an annotated document from a LangExtract dataclass.

        Parameters
        ----------
        data : LXAnnotatedDocument
            LangExtract annotated document dataclass instance.

        Returns
        -------
        AnnotatedDocument
            Structured annotated document model.
        """
        if not isinstance(data, LXAnnotatedDocument):
            return super().from_dataclass(data)
        extractions = (
            [Extraction.from_dataclass(item) for item in data.extractions]
            if data.extractions is not None
            else None
        )
        tokenized_text = (
            TokenizedText.from_dataclass(data.tokenized_text)
            if data.tokenized_text is not None
            else None
        )
        return cls(
            document_id=data.document_id,
            extractions=extractions,
            text=data.text,
            tokenized_text=tokenized_text,
        )


class Document(StructureBase):
    """Store extraction results for a document.

    Attributes
    ----------
    text : str
        Raw text representation for the document.
    document_id : str | None
        Identifier for the source document.
    additional_context : str | None
        Additional context to supplement prompt instructions.
    tokenized_text : TokenizedText | None
        Tokenized representation of the document text.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``Document`` dataclass.
    from_dataclass(data)
        Create a document from a LangExtract dataclass.
    """

    text: str = spec_field(
        "text",
        allow_null=False,
        description="Raw text representation for the document.",
    )
    document_id: str | None = spec_field(
        "document_id",
        description="Identifier for the source document.",
        allow_null=True,
    )
    additional_context: str | None = spec_field(
        "additional_context",
        description="Additional context to supplement prompt instructions.",
        allow_null=True,
    )
    tokenized_text: TokenizedText | None = spec_field(
        "tokenized_text",
        description="Tokenized representation of the document text.",
        allow_null=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """Populate default identifiers and tokenized text after validation."""
        if self.document_id is None:
            self.document_id = f"doc_{uuid.uuid4().hex[:8]}"
        if self.tokenized_text is None and self.text:
            tokenized = LXtokenizer.tokenize(self.text)
            self.tokenized_text = TokenizedText.from_dataclass(tokenized)

    def to_dataclass(self) -> LXDocument:
        """Convert to LangExtract Document dataclass.

        Returns
        -------
        LXDocument
            LangExtract document dataclass instance.
        """
        lx_doc = LXDocument(
            text=self.text,
            document_id=self.document_id,
            additional_context=self.additional_context,
        )
        if self.tokenized_text is not None:
            lx_doc.tokenized_text = self.tokenized_text.to_dataclass()
        return lx_doc

    @classmethod
    def from_dataclass(cls, data: LXDocument) -> "Document":
        """Create a document from a LangExtract dataclass.

        Parameters
        ----------
        data : LXDocument
            LangExtract document dataclass instance.

        Returns
        -------
        Document
            Structured document model.
        """
        if not isinstance(data, LXDocument):
            return super().from_dataclass(data)
        tokenized_text = (
            TokenizedText.from_dataclass(data.tokenized_text)
            if data.tokenized_text is not None
            else None
        )
        return cls(
            text=data.text,
            document_id=data.document_id,
            additional_context=data.additional_context,
            tokenized_text=tokenized_text,
        )


__all__ = [
    "AnnotatedDocument",
    "Document",
    "ExampleData",
    "Extraction",
]

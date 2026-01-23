"""Structured extraction result models."""

from __future__ import annotations

from typing import Any, Sequence
import uuid
from enum import Enum, IntEnum
from langextract.core import format_handler as lx_format_handler
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
    from_dataclass(data)
        Create a CharInterval from a LangExtract dataclass.
    """

    start_pos: int | None = spec_field(
        "start_pos",
        description="The starting position of the interval (inclusive).",
    )
    end_pos: int | None = spec_field(
        "end_pos",
        description="The ending position of the interval (exclusive).",
    )

    def to_dataclass(self) -> LXCharInterval:
        """Convert to LangExtract CharInterval dataclass.

        Returns
        -------
        LXCharInterval
            LangExtract character interval dataclass instance.
        """
        return LXCharInterval(
            start_pos=self.start_pos,
            end_pos=self.end_pos,
        )

    @classmethod
    def from_dataclass(cls, data: LXCharInterval) -> "CharInterval":
        """Create a CharInterval from a LangExtract dataclass.

        Parameters
        ----------
        data : LXCharInterval
            LangExtract CharInterval dataclass instance.

        Returns
        -------
        CharInterval
            Structured character interval model.
        """
        return cls(
            start_pos=data.start_pos,
            end_pos=data.end_pos,
        )


class AlignmentStatus(Enum):
    """Represent alignment status values for extracted items.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``AlignmentStatus`` dataclass.
    from_dataclass(data)
        Create an AlignmentStatus from a LangExtract dataclass.
    """

    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"

    def to_dataclass(self) -> LXAlignmentStatus:
        """Convert to LangExtract AlignmentStatus dataclass.

        Returns
        -------
        LXAlignmentStatus
            LangExtract alignment status dataclass instance.
        """
        return LXAlignmentStatus(self.value)

    @classmethod
    def from_dataclass(cls, data: LXAlignmentStatus) -> "AlignmentStatus":
        """Create an AlignmentStatus from a LangExtract dataclass.

        Parameters
        ----------
        data : LXAlignmentStatus
            LangExtract alignment status dataclass instance.

        Returns
        -------
        AlignmentStatus
            Structured alignment status value.
        """
        return cls(data.value)


class TokenCharInterval(StructureBase):
    """Represents an interval over characters in tokenized text.

    The interval is defined by a start position (inclusive) and an end position
    (exclusive).

    Attributes
    ----------
      start_pos: The starting position of the interval (inclusive).
      end_pos: The ending position of the interval (exclusive).

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``CharInterval`` dataclass.
    from_dataclass(data)
        Create a TokenCharInterval from a LangExtract dataclass.
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

    def to_dataclass(self) -> LXtokenizer.CharInterval:
        """Convert to LangExtract CharInterval dataclass.

        Returns
        -------
        LXtokenizer.CharInterval
            LangExtract character interval dataclass instance.
        """
        return LXtokenizer.CharInterval(
            start_pos=self.start_pos,
            end_pos=self.end_pos,
        )

    @classmethod
    def from_dataclass(cls, data: LXtokenizer.CharInterval) -> "TokenCharInterval":
        """Create a TokenCharInterval from a LangExtract dataclass.

        Parameters
        ----------
        data : LXtokenizer.CharInterval
            LangExtract CharInterval dataclass instance.

        Returns
        -------
        TokenCharInterval
            Structured token character interval model.
        """
        return cls(
            start_pos=data.start_pos,
            end_pos=data.end_pos,
        )


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
    from_dataclass(data)
        Create a TokenInterval from a LangExtract dataclass.
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
        """Convert to LangExtract TokenInterval dataclass.

        Returns
        -------
        LXtokenizer.TokenInterval
            LangExtract token interval dataclass instance.
        """
        return LXtokenizer.TokenInterval(
            start_index=self.start_index,
            end_index=self.end_index,
        )

    @classmethod
    def from_dataclass(cls, data: LXtokenizer.TokenInterval) -> "TokenInterval":
        """Create a TokenInterval from a LangExtract dataclass.

        Parameters
        ----------
        data : LXtokenizer.TokenInterval
            LangExtract TokenInterval dataclass instance.

        Returns
        -------
        TokenInterval
            Structured token interval model.
        """
        return cls(
            start_index=data.start_index,
            end_index=data.end_index,
        )


class TokenType(IntEnum):
    """Enumeration of token types produced during tokenization.

    Attributes
    ----------
      WORD: Represents an alphabetical word token.
      NUMBER: Represents a numeric token.
      PUNCTUATION: Represents punctuation characters.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``TokenType`` dataclass.
    from_dataclass(data)
        Create a TokenType from a LangExtract dataclass.
    """

    WORD = 0
    NUMBER = 1
    PUNCTUATION = 2

    def to_dataclass(self) -> LXtokenizer.TokenType:
        """Convert to LangExtract TokenType dataclass.

        Returns
        -------
        LXtokenizer.TokenType
            LangExtract token type dataclass instance.
        """
        return LXtokenizer.TokenType(self.value)

    @classmethod
    def from_dataclass(cls, data: LXtokenizer.TokenType) -> "TokenType":
        """Create a TokenType from a LangExtract dataclass.

        Parameters
        ----------
        data : LXtokenizer.TokenType
            LangExtract token type dataclass instance.

        Returns
        -------
        TokenType
            Structured token type value.
        """
        return cls(data.value)


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

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``Token`` dataclass.
    from_dataclass(data)
        Create a Token from a LangExtract dataclass.
    from_dataclass_list(data)
        Create structured tokens from LangExtract dataclasses.
    to_dataclass_list(data)
        Convert structured tokens to LangExtract dataclasses.
    """

    index: int = spec_field(
        "index",
        description="The position of the token in the sequence of tokens.",
    )
    token_type: TokenType = spec_field(
        "token_type",
        description="The type of the token, as defined by TokenType.",
    )
    char_interval: TokenCharInterval | None = spec_field(
        "char_interval",
        description="The character interval within the original text that this token spans.",
        allow_null=True,
    )
    first_token_after_newline: bool = spec_field(
        "first_token_after_newline",
        description="True if the token immediately follows a newline or carriage return.",
        default=False,
    )

    def to_dataclass(self) -> LXtokenizer.Token:
        """Convert to LangExtract Token dataclass.

        Returns
        -------
        LXtokenizer.Token
            LangExtract token dataclass instance.
        """
        token = LXtokenizer.Token(
            index=self.index,
            token_type=LXtokenizer.TokenType(self.token_type),
            first_token_after_newline=self.first_token_after_newline,
        )
        if self.char_interval is not None:
            token.char_interval = self.char_interval.to_dataclass()
        return token

    @classmethod
    def from_dataclass(cls, data: LXtokenizer.Token) -> "Token":
        """Create a Token from a LangExtract dataclass.

        Parameters
        ----------
        data : LXtokenizer.Token
            LangExtract token dataclass instance.

        Returns
        -------
        Token
            Structured token model.
        """
        char_interval = (
            TokenCharInterval.from_dataclass(data.char_interval)
            if data.char_interval is not None
            else None
        )
        return cls(
            index=data.index,
            token_type=TokenType.from_dataclass(data.token_type),
            char_interval=char_interval,
            first_token_after_newline=data.first_token_after_newline,
        )

    @staticmethod
    def from_dataclass_list(data: list[LXtokenizer.Token]) -> list["Token"]:
        """Create a list of Tokens from a list of LangExtract dataclasses.

        Parameters
        ----------
        data : list[LXtokenizer.Token]
            List of LangExtract token dataclass instances.

        Returns
        -------
        list[Token]
            List of structured token models.
        """
        return [Token.from_dataclass(item) for item in data]

    @staticmethod
    def to_dataclass_list(data: list["Token"]) -> list[LXtokenizer.Token]:
        """Convert a list of Tokens to LangExtract Token dataclasses.

        Parameters
        ----------
        data : list[Token]
            List of structured token models.

        Returns
        -------
        list[LXtokenizer.Token]
            List of LangExtract token dataclass instances.
        """
        return [item.to_dataclass() for item in data]


class TokenizedText(StructureBase):
    """Holds the result of tokenizing a text string.

    Attributes
    ----------
      text: The text that was tokenized. For UnicodeTokenizer, this is
        NOT normalized to NFC (to preserve indices).
      tokens: A list of Token objects extracted from the text.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``TokenizedText`` dataclass.
    from_dataclass(data)
        Create a TokenizedText from a LangExtract dataclass.
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
        """Convert to LangExtract TokenizedText dataclass.

        Returns
        -------
        LXtokenizer.TokenizedText
            LangExtract tokenized text dataclass instance.
        """
        return LXtokenizer.TokenizedText(
            text=self.text,
            tokens=Token.to_dataclass_list(self.tokens),
        )

    @classmethod
    def from_dataclass(cls, data: LXtokenizer.TokenizedText) -> "TokenizedText":
        """Create a TokenizedText from a LangExtract dataclass.

        Parameters
        ----------
        data : LXtokenizer.TokenizedText
            LangExtract TokenizedText dataclass instance.

        Returns
        -------
        TokenizedText
            Structured tokenized text model.
        """
        return cls(
            text=data.text,
            tokens=Token.from_dataclass_list(data.tokens),
        )


class AttributeStructure(StructureBase):
    """Represent an extraction attribute as a key/value pair.

    Attributes
    ----------
    key : str
        Attribute key.
    value : str | int | float | dict | list | None
        Attribute value.

    Methods
    -------
    to_pair()
        Convert the attribute to a tuple of ``(key, value)``.
    from_pair(key, value)
        Build an attribute from a key/value pair.
    """

    key: str = spec_field(
        "key",
        allow_null=False,
        description="Attribute key.",
    )
    value: lx_format_handler.ExtractionValueType = spec_field(
        "value",
        allow_null=True,
        description="Attribute value.",
    )

    def to_pair(self) -> tuple[str, lx_format_handler.ExtractionValueType]:
        """Convert the attribute to a key/value pair.

        Returns
        -------
        tuple[str, str | int | float | dict | list | None]
            Tuple containing the attribute key and value.
        """
        return self.key, self.value

    @classmethod
    def from_pair(
        cls, key: str, value: lx_format_handler.ExtractionValueType
    ) -> "AttributeStructure":
        """Build an attribute from a key/value pair.

        Parameters
        ----------
        key : str
            Attribute key.
        value : str | int | float | dict | list | None
            Attribute value to store.

        Returns
        -------
        AttributeStructure
            Structured attribute instance.
        """
        return cls(key=key, value=value)


def _attributes_to_dict(
    attributes: list[AttributeStructure] | None,
) -> dict[str, Any] | None:
    """Convert structured attributes to a dictionary.

    Parameters
    ----------
    attributes : list[AttributeStructure] or None
        Structured attributes to convert.

    Returns
    -------
    dict[str, Any] or None
        Mapping of attribute keys to values.
    """
    if attributes is None:
        return None
    return {attribute.key: attribute.value for attribute in attributes}


def _attributes_from_dict(
    attributes: dict[str, Any] | None,
) -> list[AttributeStructure] | None:
    """Convert an attribute dictionary into structured attributes.

    Parameters
    ----------
    attributes : dict[str, Any] or None
        Attributes mapping to convert.

    Returns
    -------
    list[AttributeStructure] or None
        Structured attribute list.
    """
    if attributes is None:
        return None
    return [
        AttributeStructure.from_pair(key, value) for key, value in attributes.items()
    ]


class ExtractionStructure(StructureBase):
    """Represent a single extraction from a document.

    Attributes
    ----------
    extraction_class : str
        Label or class assigned to the extracted item.
    extraction_text : str
        Raw text captured for the extracted item.
    description : str | None
        Optional description of the extracted item.
    attributes : list[AttributeStructure] | None
        Additional attributes attached to the item.
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
    to_dataclass_list(data)
        Convert structured extractions to LangExtract dataclasses.
    from_dataclass(data)
        Create an extraction from a LangExtract dataclass.
    from_dataclass_list(data)
        Create structured extractions from LangExtract dataclasses.
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
    description: str | None = spec_field(
        "description",
        allow_null=True,
        description="Optional description of the extracted item.",
    )
    attributes: list[AttributeStructure] | None = spec_field(
        "attributes",
        default=None,
        description="Additional attributes attached to the item.",
    )
    char_interval: CharInterval | None = spec_field(
        "char_interval",
        allow_null=True,
        description="Character interval of the extracted item in the source text.",
    )
    alignment_status: AlignmentStatus | None = spec_field(
        "alignment_status",
        allow_null=True,
        description="Alignment status of the extracted item.",
    )
    extraction_index: int | None = spec_field(
        "extraction_index",
        description="Index of the extraction in the list of extractions.",
        allow_null=True,
    )
    group_index: int | None = spec_field(
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
            self.char_interval.to_dataclass()
            if self.char_interval is not None
            else None
        )
        alignment_status = (
            self.alignment_status.to_dataclass()
            if self.alignment_status is not None
            else None
        )
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
            attributes=_attributes_to_dict(self.attributes),
            token_interval=token_interval,
        )

    @staticmethod
    def to_dataclass_list(
        data: Sequence["ExtractionStructure"],
    ) -> list[LXExtraction]:
        """Convert a list of Extractions to LangExtract Extraction dataclasses.

        Parameters
        ----------
        data : Sequence[ExtractionStructure]
            List of structured extraction models.

        Returns
        -------
        list[LXExtraction]
            List of LangExtract extraction dataclass instances.
        """
        return [item.to_dataclass() for item in data]

    @classmethod
    def from_dataclass(cls, data: LXExtraction) -> "ExtractionStructure":
        """Create an extraction from a LangExtract dataclass.

        Parameters
        ----------
        data : LXExtraction
            LangExtract extraction dataclass instance.

        Returns
        -------
        ExtractionStructure
            Structured extraction model.
        """
        char_interval = (
            CharInterval.from_dataclass(data.char_interval)
            if data.char_interval is not None
            else None
        )
        alignment_status = (
            AlignmentStatus.from_dataclass(data.alignment_status)
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
            attributes=_attributes_from_dict(data.attributes),
            token_interval=token_interval,
        )

    @staticmethod
    def from_dataclass_list(
        data: list[LXExtraction] | None,
    ) -> list["ExtractionStructure"]:
        """Create a list of extractions from a list of LangExtract dataclasses.

        Parameters
        ----------
        data : list[LXExtraction]
            List of LangExtract extraction dataclass instances.

        Returns
        -------
        list[ExtractionStructure]
            List of structured extraction models.
        """
        if data is None:
            return []
        return [ExtractionStructure.from_dataclass(item) for item in data]


class ExampleDataStructure(StructureBase):
    """Represent example data for structured prompting.

    Attributes
    ----------
    text : str
        Raw text for the example.
    extractions : list[ExtractionStructure]
        Extractions associated with the text. Default is an empty list.

    Methods
    -------
    to_dataclass()
        Convert to a LangExtract ``ExampleData`` dataclass.
    to_dataclass_list(data)
        Convert structured example data to LangExtract dataclasses.
    from_dataclass(data)
        Create example data from a LangExtract dataclass.
    from_dataclass_list(data)
        Create structured examples from LangExtract dataclasses.
    """

    text: str = spec_field(
        "text",
        allow_null=False,
        description="Raw text for the example.",
    )
    extractions: list[ExtractionStructure] = spec_field(
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
        return LXExampleData(
            text=self.text,
            extractions=ExtractionStructure.to_dataclass_list(self.extractions),
        )

    @staticmethod
    def to_dataclass_list(
        data: Sequence["ExampleDataStructure"],
    ) -> list[LXExampleData]:
        """Convert structured examples to LangExtract dataclasses.

        Parameters
        ----------
        data : Sequence[ExampleDataStructure]
            List of structured example data models.

        Returns
        -------
        list[LXExampleData]
            List of LangExtract example dataclass instances.
        """
        return [item.to_dataclass() for item in data]

    @classmethod
    def from_dataclass(cls, data: LXExampleData) -> "ExampleDataStructure":
        """Create example data from a LangExtract dataclass.

        Parameters
        ----------
        data : LXExampleData
            LangExtract example dataclass instance.

        Returns
        -------
        ExampleDataStructure
            Structured example data model.
        """
        extractions = ExtractionStructure.from_dataclass_list(data.extractions)
        return cls(text=data.text, extractions=extractions)

    @staticmethod
    def from_dataclass_list(
        data: list[LXExampleData] | None,
    ) -> list["ExampleDataStructure"]:
        """Create structured examples from LangExtract dataclasses.

        Parameters
        ----------
        data : list[LXExampleData] or None
            List of LangExtract example dataclass instances.

        Returns
        -------
        list[ExampleDataStructure]
            List of structured example data models.
        """
        if data is None:
            return []
        return [ExampleDataStructure.from_dataclass(item) for item in data]


class AnnotatedDocumentStructure(StructureBase):
    """Represent a document annotated with extractions.

    Attributes
    ----------
    document_id : str | None
        Identifier for the document.
    extractions : list[ExtractionStructure] | None
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
    extractions: list[ExtractionStructure] | None = spec_field(
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
            ExtractionStructure.to_dataclass_list(self.extractions)
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
    def from_dataclass(cls, data: LXAnnotatedDocument) -> "AnnotatedDocumentStructure":
        """Create an annotated document from a LangExtract dataclass.

        Parameters
        ----------
        data : LXAnnotatedDocument
            LangExtract annotated document dataclass instance.

        Returns
        -------
        AnnotatedDocumentStructure
            Structured annotated document model.
        """
        extractions = (
            ExtractionStructure.from_dataclass_list(data.extractions)
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


class DocumentStructure(StructureBase):
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
    to_dataclass_list(data)
        Convert structured documents to LangExtract dataclasses.
    from_dataclass(data)
        Create a document from a LangExtract dataclass.
    from_dataclass_list(data)
        Create structured documents from LangExtract dataclasses.
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

    @staticmethod
    def to_dataclass_list(
        data: Sequence["DocumentStructure"],
    ) -> list[LXDocument]:
        """Convert structured documents to LangExtract dataclasses.

        Parameters
        ----------
        data : Sequence[DocumentStructure]
            List of structured document models.

        Returns
        -------
        list[LXDocument]
            List of LangExtract document dataclass instances.
        """
        return [item.to_dataclass() for item in data]

    @classmethod
    def from_dataclass(cls, data: LXDocument) -> "DocumentStructure":
        """Create a document from a LangExtract dataclass.

        Parameters
        ----------
        data : LXDocument
            LangExtract document dataclass instance.

        Returns
        -------
        DocumentStructure
            Structured document model.
        """
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

    @staticmethod
    def from_dataclass_list(
        data: list[LXDocument] | None,
    ) -> list["DocumentStructure"]:
        """Create structured documents from LangExtract dataclasses.

        Parameters
        ----------
        data : list[LXDocument] or None
            List of LangExtract document dataclass instances.

        Returns
        -------
        list[DocumentStructure]
            List of structured document models.
        """
        if data is None:
            return []
        return [DocumentStructure.from_dataclass(item) for item in data]


class DocumentExtractorConfig(StructureBase):
    """Configuration settings for the extractor.

    Attributes
    ----------
    name : str
        Name used to store and reuse extractor configurations.
    prompt_description : str
        Prompt description used by LangExtract.
    extraction_classes : list[str]
        List of extraction classes to be extracted.
    examples : list[ExampleDataStructure]
        Example payloads supplied to LangExtract.

    Methods
    -------
    to_json()
        Return a JSON-compatible dict representation.
    to_json_file(filepath)
        Write serialized JSON data to a file path.
    """

    name: str = spec_field(
        "name",
        allow_null=False,
        description="Name used to store and reuse extractor configurations.",
        examples=["invoice_entity_extractor"],
    )
    prompt_description: str = spec_field(
        "prompt_description",
        allow_null=False,
        description="Prompt description used by LangExtract.",
        examples=[
            "Extract characters, emotions, and relationships in order of appearance. "
            "Use exact text for extractions. Do not paraphrase or overlap entities. "
            "Provide meaningful attributes for each entity to add context."
        ],
    )
    extraction_classes: list[str] = spec_field(
        "extraction_classes",
        description="List of extraction classes to be extracted.",
        default_factory=list,
        examples=[["character", "emotion", "relationship"]],
    )
    examples: list[ExampleDataStructure] = spec_field(
        "examples",
        description="Example payloads supplied to LangExtract.",
        default_factory=list,
        examples=[
            [
                ExampleDataStructure(
                    text=(
                        "ROMEO. But soft! What light through yonder window breaks? "
                        "It is the east, and Juliet is the sun."
                    ),
                    extractions=[
                        ExtractionStructure(
                            extraction_class="character",
                            extraction_text="ROMEO",
                            attributes=[
                                AttributeStructure(
                                    key="emotional_state",
                                    value="wonder",
                                )
                            ],
                        ),
                        ExtractionStructure(
                            extraction_class="emotion",
                            extraction_text="But soft!",
                            attributes=[
                                AttributeStructure(
                                    key="feeling",
                                    value="gentle awe",
                                )
                            ],
                        ),
                        ExtractionStructure(
                            extraction_class="relationship",
                            extraction_text="Juliet is the sun",
                            attributes=[
                                AttributeStructure(
                                    key="type",
                                    value="metaphor",
                                )
                            ],
                        ),
                    ],
                )
            ]
        ],
    )


__all__ = [
    "AnnotatedDocumentStructure",
    "AttributeStructure",
    "DocumentStructure",
    "ExampleDataStructure",
    "ExtractionStructure",
    "DocumentExtractorConfig",
]

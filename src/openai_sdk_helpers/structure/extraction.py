"""Structured extraction result models."""

from __future__ import annotations

from typing import Any
import uuid
from enum import Enum, IntEnum
from langextract.core.data import (
    AnnotatedDocument as LXAnnotatedDocument,
    Document as LXDocument,
)

from langextract.core import tokenizer as LXtokenizer
from .base import StructureBase, spec_field


class CharInterval(StructureBase):
    """Class for representing a character interval.

    Attributes
    ----------
      start_pos: The starting position of the interval (inclusive).
      end_pos: The ending position of the interval (exclusive).
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
        """Convert to LangExtract CharInterval dataclass."""
        return LXtokenizer.CharInterval(
            start_pos=self.start_pos,
            end_pos=self.end_pos,
        )


class AlignmentStatus(Enum):
    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"


class TokenInterval:
    """Represents an interval over tokens in tokenized text.

    The interval is defined by a start index (inclusive) and an end index
    (exclusive).

    Attributes
    ----------
      start_index: The index of the first token in the interval.
      end_index: The index one past the last token in the interval.
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


class AnnotatedDocument(StructureBase):
    """Represent a single extracted item from a document.

    Attributes
    ----------
    extraction_class : str
        Label or class assigned to the extracted item.
    extraction_text : str
        Raw text captured for the extracted item.
    attributes : dict[str, Any]
        Additional attributes attached to the item. Default is an empty dict.
    source_span : tuple[int, int] | None
        Character span in the source document, if available.
    source_id : str | None
        Identifier for the source document, if available.

    Methods
    -------
    None
        This structure relies on ``StructureBase`` methods.
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
    attributes: dict[str, Any] = spec_field(
        "attributes",
        default_factory=dict,
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


class Document(StructureBase):
    """Store extraction results for a document.

    Attributes
    ----------
    document_id : str | None
        Identifier for the source document.
    items : list[ExtractionItem]
        Extracted items for the document.
    metrics : dict[str, Any]
        Metrics and diagnostics for the extraction. Default is an empty dict.

    Methods
    -------
    None
        This structure relies on ``StructureBase`` methods.
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

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.document_id is None:
            self.document_id = f"doc_{uuid.uuid4().hex[:8]}"
        if self.tokenized_text is None:
            _tokenized_text = LXtokenizer.tokenize(self.text)
            self.tokenized_text = TokenizedText.from_dataclass(_tokenized_text)

    def to_dataclass(self) -> LXDocument:
        """Convert to LangExtract Document dataclass."""
        lx_doc = LXDocument(
            text=self.text,
            document_id=self.document_id,
            additional_context=self.additional_context,
        )
        if self.tokenized_text is None:
            raise ValueError("tokenized_text is None, cannot convert to LXDocument.")
        lx_doc.tokenized_text = self.tokenized_text.to_dataclass()
        return lx_doc


__all__ = ["AnnotatedDocument", "Document"]

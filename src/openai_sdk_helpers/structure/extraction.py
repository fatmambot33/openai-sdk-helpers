"""Structured extraction result models."""

from __future__ import annotations

from typing import Any

from .base import StructureBase, spec_field


class ExtractionItem(StructureBase):
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
    attributes: dict[str, Any] = spec_field(
        "attributes",
        default_factory=dict,
        description="Additional attributes attached to the item.",
    )
    source_span: tuple[int, int] | None = spec_field(
        "source_span",
        description="Character offsets for the extracted item.",
    )
    source_id: str | None = spec_field(
        "source_id",
        description="Identifier for the source document.",
    )


class ExtractionResult(StructureBase):
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

    document_id: str | None = spec_field(
        "document_id",
        description="Identifier for the source document.",
    )
    items: list[ExtractionItem] = spec_field(
        "items",
        allow_null=False,
        default_factory=list,
        description="Extracted items for the document.",
    )
    metrics: dict[str, Any] = spec_field(
        "metrics",
        default_factory=dict,
        description="Metrics and diagnostics for the extraction.",
    )


__all__ = ["ExtractionItem", "ExtractionResult"]

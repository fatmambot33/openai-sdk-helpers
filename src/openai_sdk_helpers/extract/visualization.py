"""Visualization helpers for document extraction."""

from __future__ import annotations

import html
from typing import Any

from ..structure.extraction import ExtractionResult


def render_extraction_html(text: str, result: ExtractionResult) -> str:
    """Render an HTML visualization of extraction spans.

    Parameters
    ----------
    text : str
        Original document text.
    result : ExtractionResult
        Extraction result with source spans.

    Returns
    -------
    str
        HTML representation with highlighted spans.
    """
    escaped_text = html.escape(text)
    if not result.items:
        return _wrap_html(escaped_text, [])

    spans = []
    for item in result.items:
        if item.source_span is None:
            continue
        start, end = item.source_span
        spans.append((start, end, item.extraction_class))

    spans.sort(key=lambda span: span[0])
    rendered = []
    cursor = 0
    for start, end, label in spans:
        if start < cursor or start >= end:
            continue
        rendered.append(html.escape(text[cursor:start]))
        highlighted = html.escape(text[start:end])
        rendered.append(
            f"<mark data-extraction='{html.escape(str(label))}'>"
            f"{highlighted}</mark>"
        )
        cursor = end

    rendered.append(html.escape(text[cursor:]))
    return _wrap_html("".join(rendered), spans)


def _wrap_html(content: str, spans: list[tuple[int, int, Any]]) -> str:
    """Wrap the highlighted content in a minimal HTML template."""
    legend_items = "".join(
        f"<li><code>{html.escape(str(label))}</code></li>" for _, _, label in spans
    )
    legend = f"<ul>{legend_items}</ul>" if legend_items else ""
    return (
        "<html><head><style>"
        "mark{background:#fff3a3;padding:0 2px;border-radius:2px;}"
        "code{background:#f4f4f4;padding:2px 4px;border-radius:4px;}"
        "</style></head><body>"
        f"{legend}<div>{content}</div>"
        "</body></html>"
    )


__all__ = ["render_extraction_html"]

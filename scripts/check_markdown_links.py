"""Validate repository-local links and anchors in Markdown files."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_INLINE_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_EXTERNAL_SCHEMES = {"data", "http", "https", "mailto", "tel"}
_IGNORED_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".venv", "build", "dist"}


def _markdown_files(root: Path) -> list[Path]:
    """Return tracked-style Markdown paths under a repository root.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    list[Path]
        Sorted Markdown paths excluding generated and virtual-environment trees.
    """
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(part in _IGNORED_DIRECTORIES for part in path.relative_to(root).parts)
    )


def _without_fenced_code(text: str) -> list[tuple[int, str]]:
    """Return numbered Markdown lines outside fenced code blocks.

    Parameters
    ----------
    text : str
        Markdown source.

    Returns
    -------
    list[tuple[int, str]]
        One-based line numbers and source lines outside code fences.
    """
    lines: list[tuple[int, str]] = []
    fence: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            lines.append((line_number, line))
    return lines


def _slugify_heading(heading: str) -> str:
    """Approximate GitHub's Markdown heading slug generation.

    Parameters
    ----------
    heading : str
        Markdown heading text.

    Returns
    -------
    str
        Normalized anchor slug.
    """
    text = _INLINE_LINK_PATTERN.sub(r"\1", heading)
    text = _HTML_TAG_PATTERN.sub("", text)
    text = text.replace("`", "").replace("*", "").replace("_", "_")
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    text = re.sub(r"\s+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def _anchors(path: Path) -> set[str]:
    """Collect GitHub-style heading anchors for one Markdown file.

    Parameters
    ----------
    path : Path
        Markdown path.

    Returns
    -------
    set[str]
        Available anchors, including duplicate-heading suffixes.
    """
    anchors: set[str] = set()
    occurrences: defaultdict[str, int] = defaultdict(int)
    text = path.read_text(encoding="utf-8")
    for _, line in _without_fenced_code(text):
        match = _HEADING_PATTERN.match(line)
        if match is None:
            continue
        base = _slugify_heading(match.group(2))
        if not base:
            continue
        count = occurrences[base]
        anchor = base if count == 0 else f"{base}-{count}"
        occurrences[base] += 1
        anchors.add(anchor)
    return anchors


def _link_target(raw_target: str) -> str:
    """Remove optional Markdown link titles and angle brackets.

    Parameters
    ----------
    raw_target : str
        Text captured from a Markdown link destination.

    Returns
    -------
    str
        URL or path portion of the destination.
    """
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


def check_markdown_links(root: Path) -> list[str]:
    """Validate local Markdown links beneath a repository root.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    list[str]
        Human-readable validation errors.
    """
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for source in _markdown_files(root):
        text = source.read_text(encoding="utf-8")
        for line_number, line in _without_fenced_code(text):
            for match in _LINK_PATTERN.finditer(line):
                target = _link_target(match.group(1))
                parsed = urlsplit(target)
                if parsed.scheme.lower() in _EXTERNAL_SCHEMES or target.startswith("//"):
                    continue

                relative_path = Path(unquote(parsed.path)) if parsed.path else Path()
                if relative_path.is_absolute():
                    destination = root / str(relative_path).lstrip("/")
                else:
                    destination = source.parent / relative_path
                destination = destination.resolve()

                try:
                    destination.relative_to(root.resolve())
                except ValueError:
                    errors.append(
                        f"{source.relative_to(root)}:{line_number}: "
                        f"link escapes repository: {target}"
                    )
                    continue

                if not destination.exists():
                    errors.append(
                        f"{source.relative_to(root)}:{line_number}: "
                        f"missing target: {target}"
                    )
                    continue

                if parsed.fragment and destination.is_file() and destination.suffix == ".md":
                    anchor = unquote(parsed.fragment).lower()
                    available = anchor_cache.setdefault(destination, _anchors(destination))
                    if anchor not in available:
                        errors.append(
                            f"{source.relative_to(root)}:{line_number}: "
                            f"missing anchor #{parsed.fragment} in "
                            f"{destination.relative_to(root)}"
                        )
    return errors


def main() -> int:
    """Run Markdown link validation from the repository root.

    Returns
    -------
    int
        Zero when all local links are valid; one otherwise.
    """
    root = Path(__file__).resolve().parents[1]
    errors = check_markdown_links(root)
    if errors:
        print("Broken internal Markdown links:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("All internal Markdown links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

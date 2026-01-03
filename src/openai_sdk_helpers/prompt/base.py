"""Core prompt rendering utilities."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, Template

load_dotenv()
warnings.filterwarnings("ignore")


class PromptRenderer:
    """Render prompts using Jinja2 templates.

    The renderer loads templates from a base directory (defaulting to the
    ``prompt`` package directory) and exposes a rendering helper for
    injecting context values. This is the foundation for dynamic prompt
    generation across all agents.

    Examples
    --------
    Basic template rendering with base_dir:

    >>> from pathlib import Path
    >>> from openai_sdk_helpers.prompt import PromptRenderer
    >>> renderer = PromptRenderer(base_dir=Path("./templates"))
    >>> prompt = renderer.render(
    ...     "greeting.jinja",
    ...     context={"name": "Alice", "language": "English"}
    ... )
    >>> print(prompt)

    Using absolute path (no base_dir required):

    >>> renderer = PromptRenderer()
    >>> prompt = renderer.render(
    ...     "/absolute/path/to/template.jinja",
    ...     context={"name": "Bob"}
    ... )

    Using default prompt directory:

    >>> renderer = PromptRenderer()  # Uses built-in templates
    >>> prompt = renderer.render("summarizer.jinja", context={})

    Methods
    -------
    render(template_path, context)
        Render the template at ``template_path`` with the supplied context.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """Initialize the renderer with a Jinja2 environment.

        Parameters
        ----------
        base_dir : Path or None, default=None
            Base directory containing Jinja2 templates. Defaults to the
            ``prompt`` directory adjacent to this file when ``None``.

        Returns
        -------
        None
        """
        if base_dir is None:
            # Defaults to the directory containing this file, which also
            # contains the builtin prompt templates.
            self.base_dir = Path(__file__).resolve().parent
        else:
            self.base_dir = base_dir

        self._env = Environment(
            loader=FileSystemLoader(str(self.base_dir)),
            autoescape=False,  # Prompts are plain text
        )

    def render(
        self, template_path: str, context: Optional[Mapping[str, Any]] = None
    ) -> str:
        """Render a Jinja2 template with the given context.

        Parameters
        ----------
        template_path : str
            Path to the template file. Can be an absolute path or relative to
            ``base_dir``.
        context : Mapping[str, Any] or None, default=None
            Context variables passed to the template.

        Returns
        -------
        str
            Rendered prompt as a string.
        """
        path = Path(template_path)
        if path.is_absolute():
            template_path_ = path
        else:
            template_path_ = Path(self.base_dir, template_path)
        template_path_text = template_path_.read_text()
        template = Template(template_path_text)
        return template.render(context or {})


__all__ = ["PromptRenderer"]

"""Core prompt rendering implementation.

This module provides the PromptRenderer class for loading and rendering
Jinja2 templates with context variables. Templates can be loaded from a
specified directory or by absolute path.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, Template

load_dotenv()
warnings.filterwarnings("ignore")


class PromptRenderer:
    """Jinja2-based template renderer for dynamic prompt generation.

    Loads and renders Jinja2 templates from a base directory or by absolute
    path. The renderer supports variable substitution, template inheritance,
    and all standard Jinja2 features for creating dynamic prompts.

    Templates are loaded from a base directory (defaulting to the built-in
    prompt package directory) or can be specified with absolute paths.
    Autoescape is disabled by default since prompts are plain text.

    Attributes
    ----------
    base_dir : Path
        Base directory for template loading.

    Methods
    -------
    render(template_path, context=None)
        Render a Jinja2 template with the given context variables.

    Examples
    --------
    Basic template rendering with custom base directory:

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

    Using built-in templates:

    >>> renderer = PromptRenderer()  # Uses built-in templates
    >>> prompt = renderer.render("summarizer.jinja", context={})
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize the renderer with a Jinja2 environment.

        Sets up the Jinja2 environment with a FileSystemLoader pointing to
        the specified base directory. If no base directory is provided,
        defaults to the built-in prompt package directory containing
        standard templates.

        Parameters
        ----------
        base_dir : Path or None, default None
            Base directory containing Jinja2 templates. If None, uses the
            prompt package directory containing built-in templates.

        Examples
        --------
        >>> from pathlib import Path
        >>> renderer = PromptRenderer(base_dir=Path("./my_templates"))
        >>> renderer.base_dir
        PosixPath('.../my_templates')

        >>> default_renderer = PromptRenderer()
        >>> default_renderer.base_dir.name
        'prompt'
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

    def render(self, template_path: str, context: dict[str, Any] | None = None) -> str:
        """Render a Jinja2 template with the given context variables.

        Loads the template from either an absolute path or a path relative
        to the base directory. The template is rendered with the provided
        context dictionary using Jinja2's template engine.

        For security, relative paths are validated to prevent path traversal
        attacks. Absolute paths are allowed but should be used with caution
        as they bypass base directory restrictions.

        Parameters
        ----------
        template_path : str
            Path to the template file. Can be an absolute path or relative
            to base_dir.
        context : dict[str, Any] or None, default None
            Context variables to pass to the template. If None, an empty
            dictionary is used.

        Returns
        -------
        str
            Fully rendered template as a string.

        Raises
        ------
        FileNotFoundError
            If the template file does not exist at the specified path.
        InputValidationError
            If the path contains suspicious patterns or attempts to escape
            the base directory.

        Examples
        --------
        >>> renderer = PromptRenderer()
        >>> context = {"name": "Alice", "age": 30}
        >>> result = renderer.render("greeting.jinja", context)
        >>> "Alice" in result
        True

        With absolute path:

        >>> result = renderer.render(
        ...     "/path/to/template.jinja",
        ...     context={"key": "value"}
        ... )
        """
        from openai_sdk_helpers.validation import validate_safe_path

        path = Path(template_path)
        if path.is_absolute():
            # Absolute paths allowed but not validated against base_dir
            template_path_ = path
        else:
            # Relative paths validated to prevent directory traversal
            template_path_ = validate_safe_path(
                self.base_dir / template_path,
                base_dir=self.base_dir,
                field_name="template_path",
            )
        template_path_text = template_path_.read_text()
        template = Template(template_path_text)
        return template.render(context or {})


__all__ = ["PromptRenderer"]

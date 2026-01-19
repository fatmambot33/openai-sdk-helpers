"""LangExtract integration helpers.

This module provides a thin adapter around LangExtract-style extractors to
normalize how extraction results are collected and validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


class LangExtractCallable(Protocol):
    """Define callable LangExtract extractor behavior.

    Methods
    -------
    __call__
        Extract structured data from text.
    """

    def __call__(self, text: str, **kwargs: Any) -> Any:
        """Extract structured data from text.

        Parameters
        ----------
        text : str
            Source text to extract from.
        **kwargs : Any
            Extra keyword arguments forwarded to the extractor.

        Returns
        -------
        Any
            Extracted structured data.
        """


class LangExtractExtractor(Protocol):
    """Define LangExtract extractor object behavior.

    Methods
    -------
    extract
        Extract structured data from text.
    """

    def extract(self, text: str, **kwargs: Any) -> Any:
        """Extract structured data from text.

        Parameters
        ----------
        text : str
            Source text to extract from.
        **kwargs : Any
            Extra keyword arguments forwarded to the extractor.

        Returns
        -------
        Any
            Extracted structured data.
        """


@dataclass(frozen=True)
class LangExtractAdapter:
    """Adapt LangExtract extractors to a consistent interface.

    Parameters
    ----------
    extractor : LangExtractCallable | LangExtractExtractor
        Callable or object providing an ``extract`` method.

    Methods
    -------
    extract
        Extract structured data from text with the configured extractor.
    extract_to_model
        Extract structured data and validate it into a Pydantic model.
    """

    extractor: LangExtractCallable | LangExtractExtractor

    def extract(self, text: str, **kwargs: Any) -> Any:
        """Extract structured data from text.

        Parameters
        ----------
        text : str
            Source text to extract from.
        **kwargs : Any
            Extra keyword arguments forwarded to the underlying extractor.

        Returns
        -------
        Any
            Extracted structured data.

        Raises
        ------
        TypeError
            If the configured extractor cannot be called.
        """
        if hasattr(self.extractor, "extract"):
            extractor = self.extractor  # type: ignore[assignment]
            return extractor.extract(text, **kwargs)  # type: ignore[union-attr]
        if callable(self.extractor):
            return self.extractor(text, **kwargs)
        raise TypeError("LangExtract extractor must be callable or expose extract().")

    def extract_to_model(
        self,
        text: str,
        model: type[TModel],
        **kwargs: Any,
    ) -> TModel:
        """Extract structured data and validate it into a Pydantic model.

        Parameters
        ----------
        text : str
            Source text to extract from.
        model : type[BaseModel]
            Pydantic model class to validate the extracted data.
        **kwargs : Any
            Extra keyword arguments forwarded to the underlying extractor.

        Returns
        -------
        BaseModel
            Validated Pydantic model instance.
        """
        extracted = self.extract(text, **kwargs)
        return model.model_validate(extracted)


def build_langextract_adapter(
    extractor: LangExtractCallable | LangExtractExtractor | None = None,
) -> LangExtractAdapter:
    """Build a LangExtract adapter from an extractor or module defaults.

    Parameters
    ----------
    extractor : LangExtractCallable | LangExtractExtractor, optional
        Explicit extractor instance or callable. If omitted, this function
        attempts to load LangExtract and use ``langextract.extract`` or
        ``langextract.Extractor``.

    Returns
    -------
    LangExtractAdapter
        Configured LangExtract adapter.

    Raises
    ------
    ImportError
        If LangExtract cannot be imported.
    AttributeError
        If no supported extractor can be resolved.
    """
    if extractor is None:
        langextract_module = _import_langextract_module()
        if hasattr(langextract_module, "extract"):
            resolved_extractor = langextract_module.extract
        elif hasattr(langextract_module, "Extractor"):
            resolved_extractor = langextract_module.Extractor()
        else:
            raise AttributeError(
                "LangExtract module does not expose extract or Extractor."
            )
        return LangExtractAdapter(extractor=resolved_extractor)
    return LangExtractAdapter(extractor=extractor)


def _import_langextract_module() -> Any:
    """Import the LangExtract module.

    Returns
    -------
    Any
        Imported LangExtract module.

    Raises
    ------
    ImportError
        If LangExtract is not installed or cannot be imported.
    """
    import importlib

    return importlib.import_module("langextract")

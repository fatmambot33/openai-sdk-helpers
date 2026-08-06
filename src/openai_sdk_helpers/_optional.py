"""Helpers for optional dependency boundaries."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def import_optional_module(
    module_name: str,
    *,
    dependency: str,
    extra: str,
    feature: str,
) -> ModuleType:
    """Import an optional module with an actionable installation error.

    Parameters
    ----------
    module_name : str
        Fully qualified module name to import.
    dependency : str
        Distribution or top-level import name that provides the module.
    extra : str
        Package extra that installs the optional dependency.
    feature : str
        Human-readable capability name used in the error message.

    Returns
    -------
    ModuleType
        Imported module.

    Raises
    ------
    ImportError
        If the optional dependency is unavailable.
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name == dependency or missing_name.startswith(
            f"{dependency}."
        ):
            raise optional_dependency_error(
                dependency=dependency,
                extra=extra,
                feature=feature,
            ) from exc
        raise


def optional_dependency_error(
    *,
    dependency: str,
    extra: str,
    feature: str,
) -> ImportError:
    """Build an actionable missing optional dependency error.

    Parameters
    ----------
    dependency : str
        Missing dependency name.
    extra : str
        Package extra that installs the dependency.
    feature : str
        Human-readable capability name.

    Returns
    -------
    ImportError
        Error containing the exact installation command.
    """
    return ImportError(
        f"{feature} requires the optional dependency '{dependency}'. "
        f"Install it with: pip install \"openai-sdk-helpers[{extra}]\""
    )


__all__ = ["import_optional_module", "optional_dependency_error"]

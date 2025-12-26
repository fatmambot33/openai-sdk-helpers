"""Environment helpers for corai."""

from __future__ import annotations

from pathlib import Path

DATETIME_FMT = "%Y%m%d_%H%M%S"


def get_data_path(module_name: str) -> Path:
    """Return a writable data directory for the given module name.

    Parameters
    ----------
    module_name : str
        Name of the module requesting a data directory.

    Returns
    -------
    Path
        Directory path under ``~/.corai`` specific to ``module_name``. The
        directory is created if it does not already exist.
    """
    base = Path.home() / ".corai"
    path = base / module_name
    path.mkdir(parents=True, exist_ok=True)
    return path

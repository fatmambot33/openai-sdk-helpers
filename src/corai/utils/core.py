"""Core utility helpers for corai."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, TypeVar

T = TypeVar("T")
_configured_logging = False


def ensure_list(value: Iterable[T] | T | None) -> List[T]:
    """Normalize a single item or iterable into a list.

    Parameters
    ----------
    value : Iterable[T] | T | None
        Item or iterable to wrap. ``None`` yields an empty list.

    Returns
    -------
    list[T]
        Normalized list representation of ``value``.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]  # type: ignore[list-item]


def check_filepath(
    filepath: Path | None = None, *, fullfilepath: str | None = None
) -> Path:
    """Ensure the parent directory for a file path exists.

    Parameters
    ----------
    filepath : Path | None, optional
        Path object to validate. Mutually exclusive with ``fullfilepath``.
    fullfilepath : str | None, optional
        String path to validate. Mutually exclusive with ``filepath``.

    Returns
    -------
    Path
        Path object representing the validated file path.

    Raises
    ------
    ValueError
        If neither ``filepath`` nor ``fullfilepath`` is provided.
    """
    if filepath is None and fullfilepath is None:
        raise ValueError("filepath or fullfilepath is required.")
    if fullfilepath is not None:
        target = Path(fullfilepath)
    elif filepath is not None:
        target = Path(filepath)
    else:
        raise ValueError("filepath or fullfilepath is required.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _to_jsonable(value: Any) -> Any:
    """Convert common helper types to JSON-serializable forms.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Any
        A JSON-safe representation of ``value``.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if hasattr(value, "model_dump"):
        model_dump = getattr(value, "model_dump")
        return model_dump()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return value


class customJSONEncoder(json.JSONEncoder):
    """Encode common helper types like enums and paths.

    Methods
    -------
    default(o)
        Return a JSON-serializable representation of ``o``.
    """

    def default(self, o: Any) -> Any:
        """Return a JSON-serializable representation of ``o``.

        Parameters
        ----------
        o : Any
            Object to serialize.

        Returns
        -------
        Any
            JSON-safe representation of ``o``.
        """
        return _to_jsonable(o)


class JSONSerializable:
    """Mixin for classes that can be serialized to JSON.

    Methods
    -------
    to_json()
        Return a JSON-compatible dict representation of the instance.
    to_json_file(filepath)
        Write serialized JSON data to a file path.
    """

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-compatible dict representation.

        Returns
        -------
        dict[str, Any]
            Mapping with only JSON-serializable values.
        """
        if is_dataclass(self) and not isinstance(self, type):
            return {k: _to_jsonable(v) for k, v in asdict(self).items()}
        if hasattr(self, "model_dump"):
            model_dump = getattr(self, "model_dump")
            return _to_jsonable(model_dump())
        return _to_jsonable(self.__dict__)

    def to_json_file(self, filepath: str | Path) -> str:
        """Write serialized JSON data to a file path.

        Parameters
        ----------
        filepath : str | Path
            Destination file path. Parent directories are created as needed.

        Returns
        -------
        str
            String representation of the file path written.
        """
        target = Path(filepath)
        check_filepath(fullfilepath=str(target))
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(
                self.to_json(),
                handle,
                indent=2,
                ensure_ascii=False,
                cls=customJSONEncoder,
            )
        return str(target)


def log(message: str, level: int = logging.INFO) -> None:
    """Log a message with a basic configuration.

    Parameters
    ----------
    message : str
        Message to emit.
    level : int, optional
        Logging level, by default ``logging.INFO``.
    """
    global _configured_logging
    if not _configured_logging:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        _configured_logging = True
    logging.log(level, message)


__all__ = [
    "ensure_list",
    "check_filepath",
    "JSONSerializable",
    "customJSONEncoder",
    "log",
]

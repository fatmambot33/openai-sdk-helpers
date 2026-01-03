"""Core utility helpers for openai-sdk-helpers.

This module provides foundational utility functions for type coercion,
file path validation, JSON serialization, and logging. These utilities
support consistent data handling across the package.
"""

from __future__ import annotations

import json
import logging
import ast
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar


def coerce_optional_float(value: Any) -> float | None:
    """Return a float when the provided value can be coerced, otherwise None.

    Handles float, int, and string inputs. Empty strings or None return None.

    Parameters
    ----------
    value : Any
        Value to convert into a float. Strings must be parseable as floats.

    Returns
    -------
    float or None
        Converted float value or None if the input is None.

    Raises
    ------
    ValueError
        If a non-empty string cannot be converted to a float.
    TypeError
        If the value is not a float-compatible type.

    Examples
    --------
    >>> coerce_optional_float(3.14)
    3.14
    >>> coerce_optional_float("2.5")
    2.5
    >>> coerce_optional_float(None) is None
    True
    """
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("timeout must be a float-compatible value") from exc
    raise TypeError("timeout must be a float, int, str, or None")


def coerce_optional_int(value: Any) -> int | None:
    """Return an int when the provided value can be coerced, otherwise None.

    Handles int, float (if whole number), and string inputs. Empty strings
    or None return None. Booleans are not considered valid integers.

    Parameters
    ----------
    value : Any
        Value to convert into an int. Strings must be parseable as integers.

    Returns
    -------
    int or None
        Converted integer value or None if the input is None.

    Raises
    ------
    ValueError
        If a non-empty string cannot be converted to an integer.
    TypeError
        If the value is not an int-compatible type.

    Examples
    --------
    >>> coerce_optional_int(42)
    42
    >>> coerce_optional_int("100")
    100
    >>> coerce_optional_int(3.0)
    3
    >>> coerce_optional_int(None) is None
    True
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError("max_retries must be an int-compatible value") from exc
    raise TypeError("max_retries must be an int, str, or None")


def coerce_dict(value: Any) -> dict[str, Any]:
    """Return a string-keyed dictionary built from value if possible.

    Converts Mapping objects to dictionaries. None returns an empty dict.

    Parameters
    ----------
    value : Any
        Mapping-like value to convert. None yields an empty dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of value.

    Raises
    ------
    TypeError
        If the value cannot be treated as a mapping.

    Examples
    --------
    >>> coerce_dict({"a": 1})
    {'a': 1}
    >>> coerce_dict(None)
    {}
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("extra_client_kwargs must be a mapping or None")


T = TypeVar("T")
_configured_logging = False


def ensure_list(value: Iterable[T] | T | None) -> list[T]:
    """Normalize a single item or iterable into a list.

    Converts None to empty list, tuples to lists, and wraps single
    items in a list.

    Parameters
    ----------
    value : Iterable[T] | T | None
        Item or iterable to wrap. None yields an empty list.

    Returns
    -------
    list[T]
        Normalized list representation of value.

    Examples
    --------
    >>> ensure_list(None)
    []
    >>> ensure_list(5)
    [5]
    >>> ensure_list([1, 2, 3])
    [1, 2, 3]
    >>> ensure_list(("a", "b"))
    ['a', 'b']
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

    Creates parent directories as needed. Exactly one of filepath or
    fullfilepath must be provided.

    Parameters
    ----------
    filepath : Path or None, optional
        Path object to validate. Mutually exclusive with fullfilepath.
    fullfilepath : str or None, optional
        String path to validate. Mutually exclusive with filepath.

    Returns
    -------
    Path
        Path object representing the validated file path.

    Raises
    ------
    ValueError
        If neither filepath nor fullfilepath is provided.

    Examples
    --------
    >>> from pathlib import Path
    >>> path = check_filepath(filepath=Path("/tmp/test.txt"))
    >>> isinstance(path, Path)
    True
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

    Handles Enum, Path, datetime, dataclasses, Pydantic models, dicts,
    lists, tuples, and sets.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Any
        A JSON-safe representation of value.

    Notes
    -----
    This is an internal helper function. Use coerce_jsonable for public API.
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


def coerce_jsonable(value: Any) -> Any:
    """Convert value into a JSON-serializable representation.

    Handles BaseStructure, BaseResponse, dataclasses, and other complex
    types by recursively converting them to JSON-compatible forms.

    Parameters
    ----------
    value : Any
        Object to convert into a JSON-friendly structure.

    Returns
    -------
    Any
        JSON-serializable representation of value.

    Examples
    --------
    >>> from datetime import datetime
    >>> result = coerce_jsonable({"date": datetime(2024, 1, 1)})
    >>> isinstance(result, dict)
    True
    """
    from openai_sdk_helpers.response.base import BaseResponse
    from openai_sdk_helpers.structure.base import BaseStructure

    if value is None:
        return None
    if isinstance(value, BaseStructure):
        return value.model_dump()
    if isinstance(value, BaseResponse):
        return coerce_jsonable(value.messages.to_json())
    if is_dataclass(value) and not isinstance(value, type):
        return {key: coerce_jsonable(item) for key, item in asdict(value).items()}
    coerced = _to_jsonable(value)
    try:
        json.dumps(coerced)
        return coerced
    except TypeError:
        return str(coerced)


class customJSONEncoder(json.JSONEncoder):
    """JSON encoder for common helper types like enums and paths.

    Extends json.JSONEncoder to handle Enum, Path, datetime, dataclasses,
    and Pydantic models automatically.

    Methods
    -------
    default(o)
        Return a JSON-serializable representation of o.

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>> json.dumps({"path": Path("/tmp")}, cls=customJSONEncoder)
    '{"path": "/tmp"}'
    """

    def default(self, o: Any) -> Any:
        """Return a JSON-serializable representation of o.

        Called by the json module when the default serialization fails.
        Delegates to _to_jsonable for type-specific conversions.

        Parameters
        ----------
        o : Any
            Object to serialize.

        Returns
        -------
        Any
            JSON-safe representation of o.
        """
        return _to_jsonable(o)


class JSONSerializable:
    """Mixin for classes that can be serialized to JSON.

    Provides to_json() and to_json_file() methods for any class. Works
    with dataclasses, Pydantic models, and regular classes with __dict__.

    Methods
    -------
    to_json()
        Return a JSON-compatible dict representation of the instance.
    to_json_file(filepath)
        Write serialized JSON data to a file path.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MyClass(JSONSerializable):
    ...     value: int
    >>> obj = MyClass(value=42)
    >>> obj.to_json()
    {'value': 42}
    """

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict representation.

        Automatically handles dataclasses, Pydantic models, and objects
        with __dict__ attributes.

        Returns
        -------
        dict[str, Any]
            Mapping with only JSON-serializable values.

        Examples
        --------
        >>> obj = JSONSerializable()
        >>> result = obj.to_json()
        >>> isinstance(result, dict)
        True
        """
        if is_dataclass(self) and not isinstance(self, type):
            return {k: _to_jsonable(v) for k, v in asdict(self).items()}
        if hasattr(self, "model_dump"):
            model_dump = getattr(self, "model_dump")
            return _to_jsonable(model_dump())
        return _to_jsonable(self.__dict__)

    def to_json_file(self, filepath: str | Path) -> str:
        """Write serialized JSON data to a file path.

        Creates parent directories as needed. Uses customJSONEncoder for
        handling special types.

        Parameters
        ----------
        filepath : str | Path
            Destination file path. Parent directories are created as needed.

        Returns
        -------
        str
            String representation of the file path written.

        Examples
        --------
        >>> obj = JSONSerializable()
        >>> path = obj.to_json_file("/tmp/output.json")  # doctest: +SKIP
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

    Configures logging on first use with a simple timestamp format.
    Subsequent calls use the existing configuration.

    Parameters
    ----------
    message : str
        Message to emit.
    level : int, optional
        Logging level (e.g., logging.INFO, logging.WARNING), by default
        logging.INFO.

    Examples
    --------
    >>> import logging
    >>> log("Test message", level=logging.INFO)  # doctest: +SKIP
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
    "coerce_jsonable",
    "JSONSerializable",
    "customJSONEncoder",
    "log",
]

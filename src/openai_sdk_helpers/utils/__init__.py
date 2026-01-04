"""Utility helpers for openai-sdk-helpers.

This package provides common utility functions for type coercion, file
handling, JSON serialization, logging, and OpenAI settings construction.
These utilities are used throughout the openai_sdk_helpers package.

Functions
---------
ensure_list(value)
    Normalize a single item or iterable into a list.
check_filepath(filepath, fullfilepath)
    Ensure the parent directory for a file path exists.
coerce_optional_float(value)
    Convert a value to float or None.
coerce_optional_int(value)
    Convert a value to int or None.
coerce_dict(value)
    Convert a value to a string-keyed dictionary.
coerce_jsonable(value)
    Convert a value into a JSON-serializable representation.
log(message, level)
    Log a message with basic configuration.
build_openai_settings(**kwargs)
    Build OpenAI settings from environment with validation.

Classes
-------
JSONSerializable
    Mixin for classes that can be serialized to JSON.
customJSONEncoder
    JSON encoder for common helper types like enums and paths.
"""

from __future__ import annotations

from .core import (
    JSONSerializable,
    build_openai_settings,
    check_filepath,
    coerce_jsonable,
    coerce_dict,
    coerce_optional_float,
    coerce_optional_int,
    customJSONEncoder,
    ensure_list,
    log,
)

__all__ = [
    "ensure_list",
    "check_filepath",
    "coerce_optional_float",
    "coerce_optional_int",
    "coerce_dict",
    "coerce_jsonable",
    "JSONSerializable",
    "customJSONEncoder",
    "log",
    "build_openai_settings",
]

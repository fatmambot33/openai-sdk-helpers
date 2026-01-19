"""Utility helpers for openai-sdk-helpers.

The utils package collects cross-cutting helpers used across the project:

* Core helpers: coercion, path handling, JSON encoding, and logging basics.
* Validation: input validation helpers for strings, choices, URLs, etc.
* Concurrency: async bridging helpers.
* Output validation: JSON Schema and semantic validators.
* Instrumentation helpers: deprecation utilities.
* Encoding: base64 encoding for images and files.

Import style
------------
Public helpers are re-exported from ``openai_sdk_helpers.utils`` for a
consistent import surface. You can import from submodules when you need a
smaller surface area, but top-level imports remain stable.

Submodules
----------
coercion
    Numeric coercion helpers and list normalization.
path_utils
    File and path helpers.
json
    JSON encoding helpers and mixins for dataclasses and Pydantic models.
logging_config
    Centralized logger factory and convenience log helper.
validation
    Input validation helpers for strings, URLs, collections, and paths.
async_utils
    Async-to-sync bridging helpers.
output_validation
    JSON Schema and semantic output validation utilities.
deprecation
    Deprecation helpers and warning utilities.
encoding
    Base64 encoding helpers for images and files.
"""

from __future__ import annotations

from .coercion import (
    coerce_dict,
    coerce_optional_float,
    coerce_optional_int,
    ensure_list,
)
from .json import (
    BaseModelJSONSerializable,
    DataclassJSONSerializable,
    coerce_jsonable,
    customJSONEncoder,
    decode_module_qualname,
    encode_module_qualname,
    get_module_qualname,
    to_jsonable,
)
from .registry import RegistryBase

from .path_utils import check_filepath, ensure_directory
from openai_sdk_helpers.logging import log
from .validation import (
    validate_choice,
    validate_dict_mapping,
    validate_list_items,
    validate_max_length,
    validate_non_empty_string,
    validate_safe_path,
    validate_url_format,
)
from .async_utils import run_coroutine_thread_safe, run_coroutine_with_fallback
from .output_validation import (
    JSONSchemaValidator,
    LengthValidator,
    OutputValidator,
    SemanticValidator,
    ValidationResult,
    ValidationRule,
    validate_output,
)

from .encoding import (
    create_file_data_url,
    create_image_data_url,
    encode_file,
    encode_image,
    get_mime_type,
    is_image_file,
)
from .langextract import (
    LangExtractAdapter,
    build_langextract_adapter,
)

__all__ = [
    "ensure_list",
    "check_filepath",
    "ensure_directory",
    "coerce_optional_float",
    "coerce_optional_int",
    "coerce_dict",
    "to_jsonable",
    "coerce_jsonable",
    "DataclassJSONSerializable",
    "BaseModelJSONSerializable",
    "customJSONEncoder",
    "get_module_qualname",
    "encode_module_qualname",
    "decode_module_qualname",
    "log",
    # Validation helpers
    "validate_non_empty_string",
    "validate_max_length",
    "validate_url_format",
    "validate_dict_mapping",
    "validate_list_items",
    "validate_choice",
    "validate_safe_path",
    # Async helpers
    "run_coroutine_thread_safe",
    "run_coroutine_with_fallback",
    # Output validation
    "ValidationResult",
    "ValidationRule",
    "JSONSchemaValidator",
    "SemanticValidator",
    "LengthValidator",
    "OutputValidator",
    "validate_output",
    # Encoding
    "encode_image",
    "encode_file",
    "get_mime_type",
    "create_image_data_url",
    "create_file_data_url",
    "is_image_file",
    # LangExtract
    "LangExtractAdapter",
    "build_langextract_adapter",
    # Registry
    "RegistryBase",
]

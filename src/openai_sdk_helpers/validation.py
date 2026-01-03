"""Input validation utilities for openai-sdk-helpers.

Provides validators and validation helpers for ensuring data integrity
at API boundaries and configuration initialization.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Callable, TypeVar

from openai_sdk_helpers.errors import InputValidationError

T = TypeVar("T")
K = TypeVar("K", bound=str)
V = TypeVar("V")
U = TypeVar("U")


def validate_non_empty_string(value: str, field_name: str) -> str:
    """Validate that a string is non-empty.

    Parameters
    ----------
    value : str
        String value to validate.
    field_name : str
        Name of the field for error messages.

    Returns
    -------
    str
        The validated (stripped) string.

    Raises
    ------
    InputValidationError
        If string is empty or only whitespace.
    """
    if not isinstance(value, str):
        raise InputValidationError(
            f"{field_name} must be a string, got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        raise InputValidationError(f"{field_name} must be non-empty")
    return stripped


def validate_max_length(value: str, max_len: int, field_name: str) -> str:
    """Validate that a string doesn't exceed maximum length.

    Parameters
    ----------
    value : str
        String value to validate.
    max_len : int
        Maximum allowed length.
    field_name : str
        Name of the field for error messages.

    Returns
    -------
    str
        The validated string.

    Raises
    ------
    InputValidationError
        If string exceeds maximum length.
    """
    if len(value) > max_len:
        raise InputValidationError(
            f"{field_name} must be <= {max_len} characters, "
            f"got {len(value)} characters"
        )
    return value


def validate_url_format(url: str, field_name: str = "URL") -> str:
    """Validate that a string is a valid URL.

    Parameters
    ----------
    url : str
        URL string to validate.
    field_name : str
        Name of the field for error messages. Default is "URL".

    Returns
    -------
    str
        The validated URL.

    Raises
    ------
    InputValidationError
        If URL format is invalid.
    """
    if not url.startswith(("http://", "https://")):
        raise InputValidationError(
            f"{field_name} must start with http:// or https://, got: {url}"
        )
    return url


def validate_dict_mapping(
    mapping: Mapping[K, V],
    expected_keys: set[K],
    field_name: str,
    allow_extra: bool = False,
) -> dict[K, V]:
    """Validate that a dict contains expected keys.

    Parameters
    ----------
    mapping : Mapping
        Dictionary-like object to validate.
    expected_keys : set
        Set of required key names.
    field_name : str
        Name of the field for error messages.
    allow_extra : bool
        Whether extra keys are allowed. Default is False.

    Returns
    -------
    dict
        The validated dictionary.

    Raises
    ------
    InputValidationError
        If required keys are missing or unexpected keys present (if not allowed).
    """
    if not isinstance(mapping, Mapping):
        raise InputValidationError(
            f"{field_name} must be a dict or mapping, got {type(mapping).__name__}"
        )

    missing_keys = expected_keys - set(mapping.keys())
    if missing_keys:
        raise InputValidationError(
            f"{field_name} missing required keys: {', '.join(sorted(missing_keys))}"
        )

    if not allow_extra:
        extra_keys = set(mapping.keys()) - expected_keys
        if extra_keys:
            raise InputValidationError(
                f"{field_name} has unexpected keys: {', '.join(sorted(extra_keys))}"
            )

    return dict(mapping)


def validate_list_items(
    items: list[U],
    item_validator: Callable[[U], T],
    field_name: str,
    allow_empty: bool = False,
) -> list[T]:
    """Validate all items in a list using a validator function.

    Parameters
    ----------
    items : list
        List to validate.
    item_validator : Callable
        Function that validates individual items.
    field_name : str
        Name of the field for error messages.
    allow_empty : bool
        Whether an empty list is allowed. Default is False.

    Returns
    -------
    list
        List of validated items.

    Raises
    ------
    InputValidationError
        If list is empty when not allowed, or if any item fails validation.
    """
    if not isinstance(items, list):
        raise InputValidationError(
            f"{field_name} must be a list, got {type(items).__name__}"
        )

    if not items and not allow_empty:
        raise InputValidationError(f"{field_name} must not be empty")

    validated = []
    for i, item in enumerate(items):
        try:
            validated.append(item_validator(item))
        except (InputValidationError, ValueError) as exc:
            raise InputValidationError(f"{field_name}[{i}] is invalid: {exc}") from exc

    return validated


def validate_choice(
    value: U,
    allowed_values: set[U],
    field_name: str,
) -> U:
    """Validate that a value is one of allowed choices.

    Parameters
    ----------
    value : Any
        Value to validate.
    allowed_values : set
        Set of allowed values.
    field_name : str
        Name of the field for error messages.

    Returns
    -------
    Any
        The validated value.

    Raises
    ------
    InputValidationError
        If value is not in allowed values.
    """
    if value not in allowed_values:
        raise InputValidationError(
            f"{field_name} must be one of {', '.join(map(str, sorted(allowed_values, key=str)))}; got: {value}"
        )
    return value


def validate_safe_path(
    path: Path | str,
    base_dir: Path | None = None,
    field_name: str = "path",
) -> Path:
    """Validate that a path is safe and does not escape the base directory.

    Protects against path traversal attacks by ensuring the resolved path
    is within the base directory when provided.

    Parameters
    ----------
    path : Path or str
        Path to validate. Can be absolute or relative.
    base_dir : Path or None
        Base directory to validate against. If None, only checks for
        suspicious patterns but allows any valid path.
    field_name : str
        Name of the field for error messages. Default is "path".

    Returns
    -------
    Path
        Validated resolved path.

    Raises
    ------
    InputValidationError
        If path is invalid, contains suspicious patterns, or escapes
        the base directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> validate_safe_path(Path("./templates/file.txt"), Path("/base"))
    PosixPath('/base/templates/file.txt')
    """
    if isinstance(path, str):
        path = Path(path)

    # Check for suspicious patterns
    path_str = str(path)
    if ".." in path.parts:
        raise InputValidationError(
            f"{field_name} contains suspicious '..' pattern: {path_str}"
        )

    # Resolve to absolute path
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise InputValidationError(
            f"{field_name} cannot be resolved: {path_str}"
        ) from exc

    # If base_dir provided, ensure path is within it
    if base_dir is not None:
        try:
            base_resolved = base_dir.resolve()
            resolved.relative_to(base_resolved)
        except ValueError:
            raise InputValidationError(
                f"{field_name} escapes base directory: {path_str} "
                f"is not within {base_dir}"
            ) from None

    return resolved

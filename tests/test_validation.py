"""Tests for input validation module."""

import pytest
from typing import Any

from openai_sdk_helpers.errors import InputValidationError
from openai_sdk_helpers.validation import (
    validate_choice,
    validate_dict_mapping,
    validate_list_items,
    validate_max_length,
    validate_non_empty_string,
    validate_url_format,
)


class TestValidateNonEmptyString:
    """Test string validation."""

    def test_valid_string(self) -> None:
        """Should accept valid non-empty string."""
        result = validate_non_empty_string("hello", "test_field")
        assert result == "hello"

    def test_string_with_whitespace_stripped(self) -> None:
        """Should strip whitespace."""
        result = validate_non_empty_string("  hello  ", "test_field")
        assert result == "hello"

    def test_empty_string_raises_error(self) -> None:
        """Should reject empty string."""
        with pytest.raises(InputValidationError, match="non-empty"):
            validate_non_empty_string("", "test_field")

    def test_whitespace_only_raises_error(self) -> None:
        """Should reject whitespace-only string."""
        with pytest.raises(InputValidationError, match="non-empty"):
            validate_non_empty_string("   ", "test_field")

    def test_non_string_raises_error(self) -> None:
        """Should reject non-string values."""
        with pytest.raises(InputValidationError, match="must be a string"):
            validate_non_empty_string(123, "test_field")  # type: ignore


class TestValidateMaxLength:
    """Test max length validation."""

    def test_valid_length(self) -> None:
        """Should accept string within limit."""
        result = validate_max_length("hello", 10, "test_field")
        assert result == "hello"

    def test_exact_max_length(self) -> None:
        """Should accept string at exactly max length."""
        result = validate_max_length("hello", 5, "test_field")
        assert result == "hello"

    def test_exceeds_max_length(self) -> None:
        """Should reject string exceeding max length."""
        with pytest.raises(InputValidationError, match="<= 5 characters"):
            validate_max_length("hello!", 5, "test_field")


class TestValidateUrlFormat:
    """Test URL format validation."""

    def test_valid_https_url(self) -> None:
        """Should accept HTTPS URL."""
        url = "https://example.com"
        result = validate_url_format(url)
        assert result == url

    def test_valid_http_url(self) -> None:
        """Should accept HTTP URL."""
        url = "http://example.com"
        result = validate_url_format(url)
        assert result == url

    def test_invalid_url_no_protocol(self) -> None:
        """Should reject URL without protocol."""
        with pytest.raises(InputValidationError, match="http:// or https://"):
            validate_url_format("example.com")

    def test_invalid_url_wrong_protocol(self) -> None:
        """Should reject URL with wrong protocol."""
        with pytest.raises(InputValidationError, match="http:// or https://"):
            validate_url_format("ftp://example.com")


class TestValidateDictMapping:
    """Test dictionary validation."""

    def test_valid_dict(self) -> None:
        """Should accept dict with expected keys."""
        mapping = {"key1": "value1", "key2": "value2"}
        result = validate_dict_mapping(mapping, {"key1", "key2"}, "test_field")
        assert result == mapping

    def test_dict_with_extra_keys_disallowed(self) -> None:
        """Should reject dict with extra keys when not allowed."""
        mapping = {"key1": "value1", "key2": "value2", "extra": "value"}
        with pytest.raises(InputValidationError, match="unexpected keys"):
            validate_dict_mapping(mapping, {"key1", "key2"}, "test_field")

    def test_dict_with_extra_keys_allowed(self) -> None:
        """Should accept dict with extra keys when allowed."""
        mapping = {"key1": "value1", "key2": "value2", "extra": "value"}
        result = validate_dict_mapping(
            mapping, {"key1", "key2"}, "test_field", allow_extra=True
        )
        assert result == mapping

    def test_dict_missing_keys(self) -> None:
        """Should reject dict with missing keys."""
        mapping = {"key1": "value1"}
        with pytest.raises(InputValidationError, match="missing required keys"):
            validate_dict_mapping(mapping, {"key1", "key2"}, "test_field")

    def test_non_dict_raises_error(self) -> None:
        """Should reject non-dict values."""
        with pytest.raises(InputValidationError, match="must be a dict"):
            validate_dict_mapping([1, 2, 3], {"key"}, "test_field")  # type: ignore


class TestValidateListItems:
    """Test list item validation."""

    def test_valid_list_items(self) -> None:
        """Should validate all items in list."""

        def validator(item: Any) -> str:
            if not isinstance(item, str):
                raise ValueError("Must be string")
            return item.upper()

        result = validate_list_items(["a", "b", "c"], validator, "test_field")
        assert result == ["A", "B", "C"]

    def test_empty_list_not_allowed(self) -> None:
        """Should reject empty list when not allowed."""

        def validator(item: Any) -> str:
            return str(item)

        with pytest.raises(InputValidationError, match="must not be empty"):
            validate_list_items([], validator, "test_field", allow_empty=False)

    def test_empty_list_allowed(self) -> None:
        """Should accept empty list when allowed."""

        def validator(item: Any) -> str:
            return str(item)

        result = validate_list_items([], validator, "test_field", allow_empty=True)
        assert result == []

    def test_invalid_list_item(self) -> None:
        """Should report which item is invalid."""

        def validator(item: Any) -> str:
            if item < 0:
                raise ValueError("Must be positive")
            return str(item)

        with pytest.raises(InputValidationError, match=r"\[1\]"):
            validate_list_items([1, -1, 3], validator, "test_field")  # type: ignore

    def test_non_list_raises_error(self) -> None:
        """Should reject non-list values."""

        def validator(item: Any) -> str:
            return str(item)

        with pytest.raises(InputValidationError, match="must be a list"):
            validate_list_items("not a list", validator, "test_field")  # type: ignore


class TestValidateChoice:
    """Test choice validation."""

    def test_valid_choice(self) -> None:
        """Should accept valid choice."""
        result = validate_choice("red", {"red", "green", "blue"}, "color")
        assert result == "red"

    def test_invalid_choice(self) -> None:
        """Should reject invalid choice."""
        with pytest.raises(InputValidationError, match="must be one of"):
            validate_choice("yellow", {"red", "green", "blue"}, "color")

    def test_numeric_choices(self) -> None:
        """Should validate numeric choices."""
        result = validate_choice(1, {1, 2, 3}, "level")
        assert result == 1

        with pytest.raises(InputValidationError):
            validate_choice(4, {1, 2, 3}, "level")

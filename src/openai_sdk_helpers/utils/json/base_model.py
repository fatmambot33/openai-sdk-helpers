"""Pydantic BaseModel JSON serialization support.

This module provides BaseModelJSONSerializable for Pydantic models,
with to_json, to_json_file, from_json, from_json_file methods and
customizable _serialize_fields/_deserialize_fields hooks.
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import inspect
import logging
import ast
from typing import Any, ClassVar, TypeVar, get_args, get_origin
from pydantic import BaseModel, ConfigDict
from ...logging import log

from .utils import customJSONEncoder

P = TypeVar("P", bound="BaseModelJSONSerializable")


class BaseModelJSONSerializable(BaseModel):
    """Pydantic BaseModel subclass with JSON serialization support.

    Adds to_json(), to_json_file(path), from_json(data), from_json_file(path),
    plus overridable _serialize_fields(data) and _deserialize_fields(data) hooks.

    Methods
    -------
    to_json()
        Return a JSON-compatible dict representation.
    to_json_file(filepath)
        Write serialized JSON data to a file path.
    from_json(data)
        Create an instance from a JSON-compatible dict (class method).
    from_json_file(filepath)
        Load an instance from a JSON file (class method).
    _serialize_fields(data)
        Customize serialization (override in subclasses).
    _deserialize_fields(data)
        Customize deserialization (override in subclasses).

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class MyConfig(BaseModelJSONSerializable, BaseModel):
    ...     name: str
    ...     value: int
    >>> cfg = MyConfig(name="test", value=42)
    >>> cfg.to_json()
    {'name': 'test', 'value': 42}
    """

    @staticmethod
    def format_output(label: str, *, value: Any) -> str:
        """
        Format a label and value for string output.

        Handles None values and lists appropriately.

        Parameters
        ----------
        label : str
            Label describing the value.
        value : Any
            Value to format for display.

        Returns
        -------
        str
            Formatted string (for example ``"- Label: Value"``).
        """
        if value is None:
            return f"- {label}: None"
        if isinstance(value, list):
            formatted = ", ".join(str(v) for v in value)
            return f"- {label}: {formatted or '[]'}"
        return f"- {label}: {str(value)}"

    def __repr__(self) -> str:
        """
        Generate a string representation of the model fields.

        Returns
        -------
        str
            Formatted string for the model fields.
        """
        return "\n".join(
            [
                BaseModelJSONSerializable.format_output(field, value=value)
                for field, value in self.model_dump().items()
            ]
        )

    def __str__(self) -> str:
        """
        Generate a string representation of the model fields.

        Returns
        -------
        str
            Formatted string for the model fields.
        """
        return self.__repr__()

    def to_markdown(self) -> str:
        """
        Generate a markdown representation of the model fields.

        Returns
        -------
        str
            Formatted markdown string for the model fields.
        """
        return self.__repr__()

    @classmethod
    def _get_all_fields(cls) -> dict[Any, Any]:
        """Collect all fields from the class hierarchy including inherited ones.

        Traverses the method resolution order (MRO) to gather fields from
        all parent classes that inherit from BaseModel, ensuring inherited
        fields are included in schema generation.

        Returns
        -------
        dict[Any, Any]
            Mapping of field names to Pydantic ModelField instances.
        """
        fields = {}
        for base in reversed(cls.__mro__):  # Traverse inheritance tree
            if issubclass(base, BaseModel) and hasattr(base, "model_fields"):
                fields.update(base.model_fields)  # Merge fields from parent
        return fields

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict representation.

        Returns
        -------
        dict[str, Any]
            Serialized model data.
        """
        return self.model_dump()

    def to_json_file(self, filepath: str | Path) -> str:
        """Write serialized JSON data to a file path.

        Parameters
        ----------
        filepath : str or Path
            Path where the JSON file will be written.

        Returns
        -------
        str
            Absolute path to the written file.
        """
        from .. import check_filepath

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

    @classmethod
    def _extract_enum_class(cls, field_type: Any) -> type[Enum] | None:
        """Extract an Enum class from a field's type annotation.

        Handles direct Enum types, list[Enum], and optional Enums.

        Parameters
        ----------
        field_type : Any
            Type annotation of a field.

        Returns
        -------
        type[Enum] or None
            Enum class if found, otherwise None.
        """
        origin = get_origin(field_type)
        args = get_args(field_type)

        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            return field_type
        elif (
            origin is list
            and args
            and inspect.isclass(args[0])
            and issubclass(args[0], Enum)
        ):
            return args[0]
        elif origin is not None:
            # Handle Union types
            for arg in args:
                enum_cls = cls._extract_enum_class(arg)
                if enum_cls:
                    return enum_cls
        return None

    @classmethod
    def _build_enum_field_mapping(cls) -> dict[str, type[Enum]]:
        """Build a mapping from field names to their Enum classes.

        Used by from_json to correctly process enum values from raw API
        responses.

        Returns
        -------
        dict[str, type[Enum]]
            Mapping of field names to Enum types.
        """
        mapping: dict[str, type[Enum]] = {}

        for name, model_field in cls.model_fields.items():
            field_type = model_field.annotation
            enum_cls = cls._extract_enum_class(field_type)

            if enum_cls is not None:
                mapping[name] = enum_cls

        return mapping

    @classmethod
    def _coerce_enum_value(
        cls, field_name: str, enum_cls: type[Enum], raw_value: Any
    ) -> Enum | None:
        """Coerce a raw enum value into an Enum member.

        Parameters
        ----------
        field_name : str
            Field name being converted.
        enum_cls : type[Enum]
            Enum class to coerce into.
        raw_value : Any
            Value to coerce into an Enum member.

        Returns
        -------
        Enum or None
            Enum member when conversion succeeds, otherwise None.
        """
        if isinstance(raw_value, enum_cls):
            return raw_value
        if isinstance(raw_value, str):
            if raw_value in enum_cls._value2member_map_:
                return enum_cls(raw_value)
            if raw_value in enum_cls.__members__:
                return enum_cls.__members__[raw_value]
        log(
            message=(
                f"[{cls.__name__}] Invalid value for '{field_name}': "
                f"'{raw_value}' not in {enum_cls.__name__}"
            ),
            level=logging.WARNING,
        )
        return None

    @classmethod
    def from_json(cls: type[P], data: dict[str, Any]) -> P:
        """Construct an instance from a dictionary of raw input data.

        Particularly useful for converting data from OpenAI API tool calls
        or assistant outputs into validated structure instances. Handles
        enum value conversion automatically.

        Parameters
        ----------
        data : dict[str, Any]
            Raw input data dictionary from API response.

        Returns
        -------
        P
            Validated instance of the model class.

        Examples
        --------
        >>> raw_data = {"title": "Test", "score": 0.95}
        >>> instance = MyStructure.from_json(raw_data)
        """
        mapping = cls._build_enum_field_mapping()
        clean_data = data.copy()

        for field, enum_cls in mapping.items():
            raw_value = clean_data.get(field)

            if raw_value is None:
                continue

            # List of enum values
            if isinstance(raw_value, list):
                converted_values = [
                    converted
                    for v in raw_value
                    if (converted := cls._coerce_enum_value(field, enum_cls, v))
                    is not None
                ]
                clean_data[field] = converted_values
            else:
                enum_value = cls._coerce_enum_value(field, enum_cls, raw_value)
                if enum_value is None:
                    clean_data[field] = None
                else:
                    clean_data[field] = enum_value

        return cls(**clean_data)

    @classmethod
    def from_json_file(cls: type[P], filepath: str | Path) -> P:
        """Load an instance from a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file to load.

        Returns
        -------
        P
            New instance of the class loaded from the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        >>> instance = MyConfig.from_json_file("configuration.json")
        """
        target = Path(filepath)
        if not target.exists():
            raise FileNotFoundError(f"JSON file not found: {target}")

        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return cls.from_json(data)

    @classmethod
    def from_string(cls: type[P], arguments: str) -> P:
        """Parse tool call arguments which may not be valid JSON.

        The OpenAI API is expected to return well-formed JSON for tool arguments,
        but minor formatting issues (such as the use of single quotes) can occur.
        This helper first tries ``json.loads`` and falls back to
        ``ast.literal_eval`` for simple cases.

        Parameters
        ----------
        arguments : str
            Raw argument string from the tool call.

        Returns
        -------
        P
            Parsed model instance from the arguments.

        Raises
        ------
        ValueError
            If the arguments cannot be parsed as JSON.

        Examples
        --------
        >>> MyModel.from_string('{"key": "value"}').key
        'value'
        """
        try:
            structured_data = json.loads(arguments)

        except json.JSONDecodeError:
            try:
                structured_data = ast.literal_eval(arguments)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Invalid JSON arguments: {arguments}. "
                    f"Expected valid JSON or Python literal."
                ) from exc
        return cls.from_json(structured_data)


__all__ = ["BaseModelJSONSerializable"]

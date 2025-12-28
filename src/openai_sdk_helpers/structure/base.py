"""Base structure definitions for shared agent structures."""

from __future__ import annotations

# Standard library imports
import inspect
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    cast,
)

# Third-party imports
from pydantic import BaseModel, ConfigDict, Field
from openai.types.responses.response_text_config_param import ResponseTextConfigParam


# Internal imports

from ..utils import check_filepath, customJSONEncoder, log

T = TypeVar("T", bound="BaseStructure")
DEFAULT_DATA_PATH: Path | None = None


class BaseStructure(BaseModel):
    """Base class for defining structured output formats for OpenAI Assistants.

    This class provides Pydantic-based schema definition and serialization
    helpers that support structured output formatting.

    Methods
    -------
    assistant_format()
        Build a response format payload for Assistant APIs.
    assistant_tool_definition(name, description)
        Build a function tool definition payload for Assistant APIs.
    get_prompt(add_enum_values)
        Format structured prompt lines into a single output string.
    get_input_prompt_list(add_enum_values)
        Build a structured prompt including inherited fields.
    get_schema(force_required)
        Generate a JSON schema for the structure.
    response_format()
        Build a response format payload for chat completions.
    response_tool_definition(tool_name, tool_description)
        Build a function tool definition payload for chat completions.
    save_schema_to_file(force_required)
        Persist the schema to disk within the application data path.
    to_json()
        Serialize the structure to a JSON-compatible dictionary.
    to_json_file(filepath)
        Write the serialized payload to ``filepath``.
    from_raw_input(data)
        Construct an instance from raw assistant tool-call arguments.
    format_output(label, value)
        Format a label/value pair for console output.
    schema_overrides()
        Produce ``Field`` overrides for dynamic schema customisation.
    print()
        Return a string representation of the structure.
    console_print()
        Print the string representation to stdout.
    """

    model_config = ConfigDict(
        title="OutputStructure", use_enum_values=False, strict=True, extra="forbid"
    )
    DATA_PATH: ClassVar[Path | None] = DEFAULT_DATA_PATH
    """Optional location for saving schema files."""

    @classmethod
    def get_prompt(cls, add_enum_values: bool = True) -> str:
        """Format structured prompt lines into a single output string.

        Parameters
        ----------
        add_enum_values : bool, default=True
            Whether enum choices should be included in the prompt lines.

        Returns
        -------
        str
            Formatted prompt ready for display.
        """
        prompt_lines = cls.get_input_prompt_list(add_enum_values)
        if not prompt_lines:
            return "No structured prompt available."
        return "# Output Format\n" + "\n".join(prompt_lines)

    @classmethod
    def _get_all_fields(cls) -> dict[Any, Any]:
        """Collect all fields, including inherited ones, from the class hierarchy.

        Returns
        -------
        dict[Any, Any]
            Mapping of field names to model fields.
        """
        fields = {}
        for base in reversed(cls.__mro__):  # Traverse inheritance tree
            if issubclass(base, BaseModel) and hasattr(base, "model_fields"):
                fields.update(base.model_fields)  # Merge fields from parent
        return fields

    @classmethod
    def _get_field_prompt(
        cls, field_name: str, field, add_enum_values: bool = True
    ) -> str:
        """Return a formatted prompt line for a single field.

        Parameters
        ----------
        field_name : str
            Name of the field being processed.
        field
            Pydantic ``ModelField`` instance.
        add_enum_values : bool, default=True
            Whether enum choices should be included.

        Returns
        -------
        str
            Single line describing the field for inclusion in the prompt.
        """
        title = field.title or field_name.capitalize()
        description = field.description or f"Provide relevant {field_name}."
        type_hint = field.annotation

        # Check for enums or list of enums
        enum_cls = cls._extract_enum_class(type_hint)
        if enum_cls:
            enum_choices_str = "\n\t\t• ".join(f"{e.name}: {e.value}" for e in enum_cls)
            if add_enum_values:
                enum_prompt = f" \n\t Choose from: \n\t\t• {enum_choices_str}"
            else:
                enum_prompt = ""

            return f"- **{title}**: {description}{enum_prompt}"

        # Otherwise check normal types
        type_mapping = {
            str: f"- **{title}**: {description}",
            bool: f"- **{title}**: {description} Specify if the {title} is true or false.",
            int: f"- **{title}**: {description} Provide the relevant integer value for {title}.",
            float: f"- **{title}**: {description} Provide the relevant float value for {title}.",
        }

        return type_mapping.get(
            type_hint, f"- **{title}**: Provide the relevant {title}."
        )

    @classmethod
    def get_input_prompt_list(cls, add_enum_values: bool = True) -> list[str]:
        """Dynamically build a structured prompt including inherited fields.

        Parameters
        ----------
        add_enum_values : bool, default=True
            Whether enumeration values should be included.

        Returns
        -------
        list[str]
            Prompt lines describing each field.
        """
        prompt_lines = []
        all_fields = cls._get_all_fields()
        for field_name, field in all_fields.items():
            prompt_lines.append(
                cls._get_field_prompt(field_name, field, add_enum_values)
            )
        return prompt_lines

    @classmethod
    def assistant_tool_definition(cls, name: str, description: str) -> dict:
        """Build an assistant function tool definition for this structure.

        Parameters
        ----------
        name : str
            Name of the function tool.
        description : str
            Description of what the function tool does.

        Returns
        -------
        dict
            Assistant tool definition payload.
        """
        from .responses import assistant_tool_definition

        return assistant_tool_definition(cls, name, description)

    @classmethod
    def assistant_format(cls) -> dict:
        """Build an assistant response format definition for this structure.

        Returns
        -------
        dict
            Assistant response format definition.
        """
        from .responses import assistant_format

        return assistant_format(cls)

    @classmethod
    def response_tool_definition(cls, tool_name: str, tool_description: str) -> dict:
        """Build a chat completion tool definition for this structure.

        Parameters
        ----------
        tool_name : str
            Name of the function tool.
        tool_description : str
            Description of what the function tool does.

        Returns
        -------
        dict
            Tool definition payload for chat completions.
        """
        from .responses import response_tool_definition

        return response_tool_definition(cls, tool_name, tool_description)

    @classmethod
    def response_format(cls) -> ResponseTextConfigParam:
        """Build a chat completion response format for this structure.

        Returns
        -------
        ResponseTextConfigParam
            Response format definition.
        """
        from .responses import response_format

        return response_format(cls)

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        """Generate a JSON schema for the class.

        All object properties are marked as required to produce fully specified
        schemas. Fields with a default value of ``None`` are treated as nullable
        and gain an explicit ``null`` entry in the resulting schema.

        Parameters
        ----------
        force_required : bool, default=False
            Retained for compatibility; all schemas declare required properties.

        Returns
        -------
        dict[str, Any]
            JSON schema describing the structure.
        """
        schema = cls.model_json_schema()

        def clean_refs(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    for key in list(obj.keys()):
                        if key != "$ref":
                            obj.pop(key, None)
                for v in obj.values():
                    clean_refs(v)
            elif isinstance(obj, list):
                for item in obj:
                    clean_refs(item)
            return obj

        cleaned_schema = cast(Dict[str, Any], clean_refs(schema))

        def add_required_fields(target: dict[str, Any]) -> None:
            """Ensure every object declares its required properties."""
            properties = target.get("properties")
            if isinstance(properties, dict) and properties:
                target["required"] = sorted(properties.keys())
            for value in target.values():
                if isinstance(value, dict):
                    add_required_fields(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            add_required_fields(item)

        nullable_fields = {
            name
            for name, model_field in getattr(cls, "model_fields", {}).items()
            if getattr(model_field, "default", inspect.Signature.empty) is None
        }

        properties = cleaned_schema.get("properties", {})
        if isinstance(properties, dict) and nullable_fields:
            for field_name in nullable_fields:
                field_props = properties.get(field_name)
                if not isinstance(field_props, dict):
                    continue

                field_type = field_props.get("type")
                if isinstance(field_type, str):
                    field_props["type"] = [field_type, "null"]
                elif isinstance(field_type, list):
                    if "null" not in field_type:
                        field_type.append("null")
                else:
                    any_of = field_props.get("anyOf")
                    if isinstance(any_of, list):
                        has_null = any(
                            isinstance(item, dict) and item.get("type") == "null"
                            for item in any_of
                        )
                        if not has_null:
                            any_of.append({"type": "null"})

        add_required_fields(cleaned_schema)
        return cleaned_schema

    @classmethod
    def save_schema_to_file(cls) -> Path:
        """
        Save the generated JSON schema to a file.

        The schema is generated using :meth:`get_schema` and saved in the
        application's data path.

        Parameters
        ----------
        force_required : bool, default=False
            When ``True``, mark all object properties as required.

        Returns
        -------
        Path
            Path to the saved schema file.
        """
        schema = cls.get_schema()
        if cls.DATA_PATH is None:
            raise RuntimeError(
                "DATA_PATH is not set. Set BaseStructure.DATA_PATH before saving."
            )
        file_path = cls.DATA_PATH / f"{cls.__name__}_schema.json"
        check_filepath(file_path)
        with file_path.open("w", encoding="utf-8") as file_handle:
            json.dump(schema, file_handle, indent=2, ensure_ascii=False)
        return file_path

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize the Pydantic model instance to a JSON-compatible dictionary.

        Enum members are converted to their values. Lists and nested dictionaries
        are recursively processed.

        Returns
        -------
        dict[str, Any]
            Model instance serialized as a dictionary.
        """

        def convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, BaseStructure):
                return obj.to_json()
            if isinstance(obj, Mapping):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, Sequence) and not isinstance(
                obj, (str, bytes, bytearray)
            ):
                return [convert(item) for item in obj]
            return obj

        payload = convert(self.model_dump())

        def is_list_field(field) -> bool:
            annotation = getattr(field, "annotation", None)
            if annotation is None:
                return False

            origins_to_match = {list, List, Sequence, tuple, set}

            origin = get_origin(annotation)
            if origin in origins_to_match or annotation in origins_to_match:
                return True

            if origin is Union:
                return any(
                    get_origin(arg) in origins_to_match or arg in origins_to_match
                    for arg in get_args(annotation)
                )
            return False

        for name, field in self.__class__.model_fields.items():
            if name not in payload:
                continue
            if not is_list_field(field):
                continue
            value = payload[name]
            if value is None:
                continue
            if isinstance(value, (str, bytes, bytearray)):
                payload[name] = [value]
            elif not isinstance(value, list):
                payload[name] = [value]

        return payload

    def to_json_file(self, filepath: str) -> str:
        """Write :meth:`to_json` output to ``filepath``.

        Parameters
        ----------
        filepath : str
            Destination path for the JSON file.

        Returns
        -------
        str
            Path to the written file.
        """
        check_filepath(fullfilepath=filepath)
        with open(file=filepath, mode="w", encoding="utf-8") as f:
            json.dump(
                self.to_json(), f, ensure_ascii=False, indent=4, cls=customJSONEncoder
            )
        return filepath

    @classmethod
    def _extract_enum_class(cls, field_type: Any) -> Optional[Type[Enum]]:
        """
        Extract an Enum class from a field's type annotation.

        Handles direct Enum types, List[Enum], and Optional[Enum] (via Union).

        Parameters
        ----------
        field_type
            Type annotation of a field.

        Returns
        -------
        type[Enum] or None
            Enum class if found, otherwise ``None``.
        """
        origin = get_origin(field_type)
        args = get_args(field_type)

        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            return field_type
        elif (
            origin in {list, List}
            and args
            and inspect.isclass(args[0])
            and issubclass(args[0], Enum)
        ):
            return args[0]
        elif origin is Union:
            for arg in args:
                enum_cls = cls._extract_enum_class(arg)
                if enum_cls:
                    return enum_cls
        return None

    @classmethod
    def _build_enum_field_mapping(cls) -> dict[str, Type[Enum]]:
        """
        Build a mapping from field names to their Enum classes.

        This is used by `from_raw_input` to correctly process enum values.

        Returns
        -------
        dict[str, type[Enum]]
            Mapping of field names to Enum types.
        """
        mapping: dict[str, Type[Enum]] = {}

        for name, model_field in cls.model_fields.items():
            field_type = model_field.annotation
            enum_cls = cls._extract_enum_class(field_type)

            if enum_cls is not None:
                mapping[name] = enum_cls

        return mapping

    @classmethod
    def from_raw_input(cls: Type[T], data: dict) -> T:
        """
        Construct an instance of the class from a dictionary of raw input data.

        This method is particularly useful for converting data received from an
        OpenAI Assistant (e.g., tool call arguments) into a Pydantic model.
        It handles the conversion of string values to Enum members for fields
        typed as Enum or List[Enum]. Warnings are logged for invalid enum values.

        Parameters
        ----------
        data : dict
            Raw input data payload.

        Returns
        -------
        T
            Instance populated with the processed data.
        """
        mapping = cls._build_enum_field_mapping()
        clean_data = data.copy()

        for field, enum_cls in mapping.items():
            raw_value = clean_data.get(field)

            if raw_value is None:
                continue

            # List of enum values
            if isinstance(raw_value, list):
                converted = []
                for v in raw_value:
                    if isinstance(v, enum_cls):
                        converted.append(v)
                    elif isinstance(v, str):
                        # Check if it's a valid value
                        if v in enum_cls._value2member_map_:
                            converted.append(enum_cls(v))
                        # Check if it's a valid name
                        elif v in enum_cls.__members__:
                            converted.append(enum_cls.__members__[v])
                        else:
                            log(
                                f"[{cls.__name__}] Skipping invalid value for '{field}': '{v}' not in {enum_cls.__name__}",
                                level=logging.WARNING,
                            )
                clean_data[field] = converted

            # Single enum value
            elif (
                isinstance(raw_value, str) and raw_value in enum_cls._value2member_map_
            ):
                clean_data[field] = enum_cls(raw_value)

            elif isinstance(raw_value, enum_cls):
                # already the correct type
                continue

            else:
                log(
                    message=f"[{cls.__name__}] Invalid value for '{field}': '{raw_value}' not in {enum_cls.__name__}",
                    level=logging.WARNING,
                )
                clean_data[field] = None

        return cls(**clean_data)

    @staticmethod
    def format_output(label: str, value: Any) -> str:
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
        if not value:
            return f"- {label}: None"
        if isinstance(value, list):
            return f"- {label}: {', '.join(str(v) for v in value)}"
        return f"- {label}: {str(value)}"

    @classmethod
    def schema_overrides(cls) -> Dict[str, Any]:
        """
        Generate Pydantic ``Field`` overrides.

        Returns
        -------
        dict[str, Any]
            Mapping of field names to ``Field`` overrides.
        """
        return {}

    def print(self) -> str:
        """
        Generate a string representation of the structure.

        Returns
        -------
        str
            Formatted string for the ``logic`` field.
        """
        return "\n".join(
            [
                BaseStructure.format_output(field, value)
                for field, value in self.model_dump().items()
            ]
        )

    def console_print(self) -> None:
        """Output the result of :meth:`print` to stdout.

        Returns
        -------
        None
        """
        print(self.print())


@dataclass(frozen=True)
class SchemaOptions:
    """Options for schema generation helpers.

    Methods
    -------
    to_kwargs()
        Return keyword arguments for schema helper calls.

    Parameters
    ----------
    force_required : bool, default=False
        When ``True``, mark all object properties as required.
    """

    force_required: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for schema helper calls.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for schema helper methods.
        """
        return {"force_required": self.force_required}


def spec_field(
    name: str,
    *,
    allow_null: bool = True,
    description: str | None = None,
    **overrides: Any,
) -> Any:
    """Return a Pydantic ``Field`` with sensible defaults for nullable specs.

    Parameters
    ----------
    name : str
        Name of the field to use as the default title.
    allow_null : bool, default=True
        When ``True``, set ``None`` as the default value to allow explicit
        ``null`` in generated schemas.
    description : str or None, default=None
        Optional description to include. When ``allow_null`` is ``True``, the
        nullable hint "Return null if none apply." is appended.
    **overrides
        Additional keyword arguments forwarded to ``pydantic.Field``.

    Returns
    -------
    Any
        Pydantic ``Field`` configured with a default title and null behavior.
    """
    field_kwargs: Dict[str, Any] = {"title": name.replace("_", " ").title()}
    field_kwargs.update(overrides)

    base_description = field_kwargs.pop("description", description)

    has_default = "default" in field_kwargs
    has_default_factory = "default_factory" in field_kwargs

    if allow_null:
        if not has_default and not has_default_factory:
            field_kwargs["default"] = None
        nullable_hint = "Return null if none apply."
        if base_description:
            field_kwargs["description"] = f"{base_description} {nullable_hint}"
        else:
            field_kwargs["description"] = nullable_hint
    else:
        if not has_default and not has_default_factory:
            field_kwargs["default"] = ...
        if base_description is not None:
            field_kwargs["description"] = base_description

    return Field(**field_kwargs)

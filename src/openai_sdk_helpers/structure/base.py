"""Base classes for structured output models.

This module provides the foundational StructureBase class and utilities for
defining Pydantic-based structured output models with OpenAI-compatible schema
generation, validation, and serialization.
"""

from __future__ import annotations

# Standard library imports
import copy
import dataclasses
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Mapping,
    TypeVar,
    cast,
)

# Third-party imports
from pydantic import BaseModel, ConfigDict, Field
from openai.types.responses.response_text_config_param import ResponseTextConfigParam

# Internal imports

from ..utils import check_filepath, BaseModelJSONSerializable

T = TypeVar("T", bound="StructureBase")


def _add_required_fields(target: dict[str, Any]) -> None:
    """Ensure every object declares its required properties."""
    properties = target.get("properties")
    if isinstance(properties, dict) and properties:
        target["required"] = sorted(properties.keys())
    for value in target.values():
        if isinstance(value, dict):
            _add_required_fields(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _add_required_fields(item)


def _enforce_additional_properties(target: Any) -> None:
    """Ensure every object schema disallows additional properties."""
    if isinstance(target, dict):
        schema_type = target.get("type")
        allows_object_type = schema_type == "object" or (
            isinstance(schema_type, list)
            and "object" in schema_type
            and set(schema_type).issubset({"object", "null"})
        )
        if (allows_object_type or "properties" in target) and "$ref" not in target:
            target.setdefault("properties", {})
            target["additionalProperties"] = False
        any_of = target.get("anyOf")
        if isinstance(any_of, list):
            for entry in any_of:
                if not isinstance(entry, dict):
                    continue
                entry_type = entry.get("type")
                entry_allows_object_type = entry_type == "object" or (
                    isinstance(entry_type, list)
                    and "object" in entry_type
                    and set(entry_type).issubset({"object", "null"})
                )
                if (
                    entry_allows_object_type or "properties" in entry
                ) and "$ref" not in entry:
                    entry.setdefault("properties", {})
                    entry["additionalProperties"] = False
        for value in target.values():
            _enforce_additional_properties(value)
    elif isinstance(target, list):
        for item in target:
            _enforce_additional_properties(item)


def _build_any_value_schema(depth: int = 0) -> dict[str, Any]:
    """Return a JSON schema fragment describing a permissive JSON value.

    Parameters
    ----------
    depth : int, optional
        Current recursion depth for nested arrays. Defaults to 0.

    Returns
    -------
    dict[str, Any]
        JSON schema fragment describing a permissive value.
    """
    value_types = ["string", "integer", "number", "null"]
    any_of: list[dict[str, Any]] = [{"type": value_type} for value_type in value_types]

    any_of.append(
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )
    if depth < 1:
        any_of.append(
            {
                "type": "array",
                "items": _build_any_value_schema(depth + 1),
            }
        )

    return {"anyOf": any_of}


def _ensure_items_have_schema(target: Any) -> None:
    """Ensure array item schemas include type information."""
    if isinstance(target, dict):
        items = target.get("items")
        if isinstance(items, dict):
            if not items:
                target["items"] = _build_any_value_schema()
            else:
                _ensure_schema_has_type(items)
        for value in target.values():
            _ensure_items_have_schema(value)
    elif isinstance(target, list):
        for item in target:
            _ensure_items_have_schema(item)


def _ensure_schema_has_type(schema: dict[str, Any]) -> None:
    """Ensure a schema dictionary includes a type entry when possible."""
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        for entry in any_of:
            if isinstance(entry, dict):
                _ensure_schema_has_type(entry)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for value in properties.values():
            if isinstance(value, dict):
                _ensure_schema_has_type(value)
    items = schema.get("items")
    if isinstance(items, dict):
        _ensure_schema_has_type(items)
    if "type" in schema or "$ref" in schema:
        return
    if isinstance(any_of, list):
        inferred_types: set[str] = set()
        for entry in any_of:
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("type")
            if isinstance(entry_type, str):
                inferred_types.add(entry_type)
            elif isinstance(entry_type, list):
                inferred_types.update(
                    element for element in entry_type if isinstance(element, str)
                )
        if inferred_types:
            schema["type"] = sorted(inferred_types)
            return
    if "properties" in schema:
        schema["type"] = "object"
        schema.setdefault("additionalProperties", False)
        return
    if "items" in schema:
        schema["type"] = "array"
        return
    schema.update(_build_any_value_schema())


def _hydrate_ref_types(schema: dict[str, Any]) -> None:
    """Attach explicit types to $ref nodes when available.

    Parameters
    ----------
    schema : dict[str, Any]
        Schema dictionary to hydrate in place.
    """
    definitions = schema.get("$defs") or schema.get("definitions") or {}
    if not isinstance(definitions, dict):
        definitions = {}

    def _infer_enum_type(values: list[Any]) -> list[str] | str | None:
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
        }
        inferred: set[str] = set()
        for value in values:
            inferred_type = type_map.get(type(value))
            if inferred_type is not None:
                inferred.add(inferred_type)
        if not inferred:
            return None
        if len(inferred) == 1:
            return next(iter(inferred))
        return sorted(inferred)

    def _resolve_ref_type(ref: str) -> list[str] | str | None:
        prefixes = ("#/$defs/", "#/definitions/")
        if not ref.startswith(prefixes):
            return None
        key = ref.split("/", maxsplit=2)[-1]
        definition = definitions.get(key)
        if not isinstance(definition, dict):
            return None
        ref_type = definition.get("type")
        if isinstance(ref_type, (str, list)):
            return ref_type
        enum_values = definition.get("enum")
        if isinstance(enum_values, list):
            return _infer_enum_type(enum_values)
        return None

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if "$ref" in node and "type" not in node:
                ref_type = _resolve_ref_type(node["$ref"])
                if ref_type is not None:
                    node["type"] = ref_type
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)


class StructureBase(BaseModelJSONSerializable):
    """Base class for structured output models with schema generation.

    Provides Pydantic-based schema definition and serialization utilities
    for OpenAI-compatible structured outputs. All structured data types
    extend this class to ensure consistent validation, serialization, and
    schema generation across the package.

    Supports automatic JSON schema generation, prompt formatting, and
    conversion to/from OpenAI API formats for both Assistant and chat
    completion APIs.

    Attributes
    ----------
    model_config : ConfigDict
        Pydantic model configuration with strict validation and enum handling.


    Methods
    -------
    get_prompt(add_enum_values=True)
        Format structured prompt lines into a single output string.
    get_input_prompt_list(add_enum_values=True)
        Build a structured prompt including inherited fields.
    assistant_format()
        Build a response format payload for Assistant APIs.
    assistant_tool_definition(name, description)
        Build a function tool definition payload for Assistant APIs.
    response_format()
        Build a response format payload for chat completions.
    response_tool_definition(tool_name, tool_description)
        Build a function tool definition payload for chat completions.
    get_schema()
        Generate a JSON schema for the structure.
    save_schema_to_file()
        Persist the schema to disk within DATA_PATH.
    to_json()
        Serialize the structure to a JSON-compatible dictionary.
    to_json_file(filepath)
        Write the serialized payload to a file.
    from_raw_input(data)
        Construct an instance from raw assistant tool-call arguments.
    format_output(label, value)
        Format a label/value pair for console output.
    schema_overrides()
        Produce Field overrides for dynamic schema customization.
    print()
        Return a string representation of the structure.

    Examples
    --------
    Define a custom structure:

    >>> from openai_sdk_helpers.structure import StructureBase, spec_field
    >>> class MyOutput(StructureBase):
    ...     title: str = spec_field("title", description="The title")
    ...     score: float = spec_field("score", description="Quality score")

    Generate JSON schema:

    >>> schema = MyOutput.get_schema()
    >>> print(schema)

    Create response format for chat completions:

    >>> format_spec = MyOutput.response_format()

    Serialize instance:

    >>> instance = MyOutput(title="Test", score=0.95)
    >>> json_dict = instance.to_json()
    """

    model_config = ConfigDict(
        title=__qualname__, use_enum_values=False, strict=True, extra="forbid"
    )
    _schema_cache: ClassVar[dict[type["StructureBase"], dict[str, Any]]] = {}

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
    def get_input_prompt_list(cls, add_enums: bool = True) -> list[str]:
        """Dynamically build a structured prompt including inherited fields.

        Parameters
        ----------
        add_enums : bool, default=True
            Whether enumeration values should be included.

        Returns
        -------
        list[str]
            Prompt lines describing each field.
        """
        prompt_lines = []
        all_fields = cls._get_all_fields()
        for field_name, field in all_fields.items():
            prompt_lines.append(cls._get_field_prompt(field_name, field, add_enums))
        return prompt_lines

    @classmethod
    def assistant_tool_definition(cls, name: str, *, description: str) -> dict:
        """Build an Assistant API function tool definition for this structure.

        Creates a tool definition compatible with the OpenAI Assistant API,
        using the structure's schema as the function parameters.

        Parameters
        ----------
        name : str
            Name of the function tool.
        description : str
            Description of what the function tool does.

        Returns
        -------
        dict
            Assistant tool definition payload in OpenAI format.

        Examples
        --------
        >>> tool = MyStructure.assistant_tool_definition(
        ...     "analyze_data",
        ...     description="Analyze the provided data"
        ... )
        """
        from .responses import assistant_tool_definition

        return assistant_tool_definition(cls, name, description)

    @classmethod
    def assistant_format(cls) -> dict:
        """Build an Assistant API response format definition.

        Creates a response format specification that instructs the Assistant
        API to return structured output matching this structure's schema.

        Returns
        -------
        dict
            Assistant response format definition in OpenAI format.

        Examples
        --------
        >>> format_def = MyStructure.assistant_format()
        """
        from .responses import assistant_format

        return assistant_format(cls)

    @classmethod
    def response_tool_definition(
        cls, tool_name: str, *, tool_description: str | None
    ) -> dict:
        """Build a chat completion tool definition for this structure.

        Creates a function tool definition compatible with the chat
        completions API, using the structure's schema as parameters.

        Parameters
        ----------
        tool_name : str
            Name of the function tool.
        tool_description : str
            Description of what the function tool does.

        Returns
        -------
        dict
            Tool definition payload for chat completions API.

        Examples
        --------
        >>> tool = MyStructure.response_tool_definition(
        ...     "process_data",
        ...     tool_description="Process the input data"
        ... )
        """
        from .responses import response_tool_definition

        return response_tool_definition(cls, tool_name, tool_description)

    @classmethod
    def response_format(cls) -> ResponseTextConfigParam:
        """Build a chat completion response format for this structure.

        Creates a response format specification that instructs the chat
        completions API to return structured output matching this
        structure's schema.

        Returns
        -------
        ResponseTextConfigParam
            Response format definition for chat completions API.

        Examples
        --------
        >>> format_spec = MyStructure.response_format()
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[...],
        ...     response_format=format_spec
        ... )
        """
        from .responses import response_format

        return response_format(cls)

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        """Generate a JSON schema for this structure.

        Produces a complete JSON schema with all properties marked as
        required. Fields with default value of None are treated as nullable
        and include an explicit null type in the schema.

        Returns
        -------
        dict[str, Any]
            JSON schema describing the structure in JSON Schema format.

        Notes
        -----
        The schema generation automatically:
        - Marks all object properties as required
        - Adds null type for fields with None default
        - Cleans up $ref entries for better compatibility
        - Recursively processes nested structures
        - Caches the computed schema per class

        Examples
        --------
        >>> schema = MyStructure.get_schema()
        >>> print(json.dumps(schema, indent=2))
        """
        cached_schema = cls._schema_cache.get(cls)
        if cached_schema is not None:
            return copy.deepcopy(cached_schema)

        schema = cls.model_json_schema()

        def clean_refs(obj: Any) -> Any:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    for key in list(obj.keys()):
                        if key not in {"$ref", "type"}:
                            obj.pop(key, None)
                for v in obj.values():
                    clean_refs(v)
            elif isinstance(obj, list):
                for item in obj:
                    clean_refs(item)
            return obj

        cleaned_schema = cast(dict[str, Any], clean_refs(schema))

        cleaned_schema = cast(dict[str, Any], cleaned_schema)
        _hydrate_ref_types(cleaned_schema)
        _ensure_items_have_schema(cleaned_schema)
        _ensure_schema_has_type(cleaned_schema)

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

        _add_required_fields(cleaned_schema)
        _enforce_additional_properties(cleaned_schema)
        cls._schema_cache[cls] = cleaned_schema
        return copy.deepcopy(cleaned_schema)

    @classmethod
    def save_schema_to_file(cls, file_path: Path) -> Path:
        """Save the generated JSON schema to a file.

        Generates the schema using get_schema and saves it to the provided
        file path.

        Parameters
        ----------
        file_path : Path
            Full path (including filename) where the schema should be saved.

        Returns
        -------
        Path
            Absolute path to the saved schema file.


        Examples
        --------
        >>> MyStructure.DATA_PATH = Path("./schemas")
        >>> schema_path = MyStructure.save_schema_to_file(file_path=MyStructure.DATA_PATH / "MyStructure_schema.json")
        >>> print(schema_path)
        PosixPath('./schemas/MyStructure_schema.json')
        """
        check_filepath(file_path)
        with file_path.open("w", encoding="utf-8") as file_handle:
            json.dump(cls.get_schema(), file_handle, indent=2, ensure_ascii=False)
        return file_path

    @classmethod
    def schema_overrides(cls) -> dict[str, Any]:
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
                StructureBase.format_output(field, value=value)
                for field, value in self.model_dump().items()
            ]
        )

    @classmethod
    def from_dataclass(cls: type[T], data: Any) -> T:
        """Create an instance from a dataclass object.

        Parameters
        ----------
        data : Any
            Dataclass instance, mapping, or object with attributes to convert.
            Private attributes (prefixed with ``_``) are ignored.

        Returns
        -------
        T
            New instance of the structure populated from the dataclass.
        """

        def _filter_private(items: list[tuple[str, Any]]) -> dict[str, Any]:
            return {name: value for name, value in items if not name.startswith("_")}

        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            payload = dataclasses.asdict(data, dict_factory=_filter_private)
        elif isinstance(data, Mapping):
            payload = _filter_private(list(data.items()))
        else:
            payload = _filter_private(list(vars(data).items()))
        return cls(**payload)


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
    field_kwargs: dict[str, Any] = {"title": name.replace("_", " ").title()}
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

"""Schema helpers for document extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SchemaLike = Mapping[str, Any]


def validate_schema_dict(schema: SchemaLike) -> dict[str, Any]:
    """Validate and normalize a schema dictionary.

    Parameters
    ----------
    schema : Mapping[str, Any]
        Schema dictionary describing the expected extractions.

    Returns
    -------
    dict[str, Any]
        Normalized schema dictionary.

    Raises
    ------
    ValueError
        If the schema dictionary is missing required keys or is malformed.
    """
    if not isinstance(schema, Mapping):
        raise ValueError("Schema must be a mapping with field definitions.")

    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("Schema must include a non-empty 'fields' list.")

    normalized_fields: list[dict[str, Any]] = []
    for field in fields:
        if not isinstance(field, Mapping):
            raise ValueError("Each schema field must be a mapping.")
        name = field.get("name")
        description = field.get("description")
        if not name or not isinstance(name, str):
            raise ValueError("Each schema field must include a string 'name'.")
        if not description or not isinstance(description, str):
            raise ValueError(
                f"Schema field '{name}' must include a string 'description'."
            )
        normalized_fields.append(dict(field))

    normalized_schema = dict(schema)
    normalized_schema["fields"] = normalized_fields
    return normalized_schema


def build_prompt_from_schema(schema: SchemaLike) -> str:
    """Build a prompt string from a schema dictionary.

    Parameters
    ----------
    schema : Mapping[str, Any]
        Schema dictionary describing expected extraction fields.

    Returns
    -------
    str
        Prompt text suitable for LangExtract.

    Raises
    ------
    ValueError
        If the schema dictionary is malformed.
    """
    normalized_schema = validate_schema_dict(schema)
    title = normalized_schema.get("name", "Document extraction")
    description = normalized_schema.get(
        "description",
        "Extract structured information from the document.",
    )

    lines = [f"{title}:", description, "Fields to extract:"]
    for field in normalized_schema["fields"]:
        field_name = field["name"]
        field_type = field.get("type")
        field_required = field.get("required", False)
        required_label = "required" if field_required else "optional"
        type_label = f" ({field_type})" if field_type else ""
        field_description = field["description"]
        lines.append(
            f"- {field_name}{type_label} [{required_label}]: {field_description}"
        )

    return "\n".join(lines)


def build_examples_from_schema(schema: SchemaLike) -> list[dict[str, Any]] | None:
    """Build example payloads from a schema dictionary.

    Parameters
    ----------
    schema : Mapping[str, Any]
        Schema dictionary describing example values.

    Returns
    -------
    list[dict[str, Any]] | None
        Example payloads when available; otherwise ``None``.

    Raises
    ------
    ValueError
        If the schema dictionary is malformed.
    """
    normalized_schema = validate_schema_dict(schema)

    examples = normalized_schema.get("examples")
    if isinstance(examples, list) and examples:
        return examples

    example_fields: dict[str, Any] = {}
    for field in normalized_schema["fields"]:
        if "example" in field:
            example_fields[field["name"]] = field["example"]

    if example_fields:
        return [example_fields]

    return None


__all__ = [
    "SchemaLike",
    "build_prompt_from_schema",
    "build_examples_from_schema",
    "validate_schema_dict",
]

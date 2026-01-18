"""Module defining the ResponseConfiguration dataclass for managing OpenAI SDK responses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, Sequence, Type, TypeVar

from ..settings import OpenAISettings
from ..structure.base import StructureBase
from .base import ResponseBase
from ..tools import ToolHandlerRegistration
from ..utils.json.data_class import DataclassJSONSerializable
from ..utils.registry import RegistryBase
from ..utils.instructions import resolve_instructions_from_path

TIn = TypeVar("TIn", bound="StructureBase")
TOut = TypeVar("TOut", bound="StructureBase")


class ResponseRegistry(RegistryBase["ResponseConfiguration"]):
    """Registry for managing ResponseConfiguration instances.

    Inherits from RegistryBase to provide centralized storage and retrieval
    of response configurations, enabling reusable response specs across the application.

    Methods
    -------
    register(configuration)
        Add a configuration to the registry.
    get(name)
        Retrieve a configuration by name.
    list_names()
        Return all registered configuration names.
    clear()
        Remove all registered configurations.
    save_to_directory(path)
        Export all registered configurations to JSON files.
    load_from_directory(path, config_class)
        Load configurations from JSON files in a directory.

    Examples
    --------
    >>> registry = ResponseRegistry()
    >>> configuration = ResponseConfiguration(
    ...     name="test",
    ...     instructions="Test instructions",
    ...     tools=None,
    ...     input_structure=None,
    ...     output_structure=None
    ... )
    >>> registry.register(configuration)
    >>> retrieved = registry.get("test")
    >>> retrieved.name
    'test'
    """

    pass


def get_default_registry() -> ResponseRegistry:
    """Return the global default registry instance.

    Returns
    -------
    ResponseRegistry
        Singleton registry for application-wide configuration storage.

    Examples
    --------
    >>> registry = get_default_registry()
    >>> configuration = ResponseConfiguration(...)
    >>> registry.register(configuration)
    """
    return _default_registry


@dataclass(frozen=True, slots=True)
class ResponseConfiguration(DataclassJSONSerializable, Generic[TIn, TOut]):
    """Represent an immutable configuration describing input and output structures.

    Encapsulate all metadata required to define how a request is interpreted and
    how a response is structured, while enforcing strict type and runtime safety.
    Inherit from DataclassJSONSerializable to support serialization to JSON format.

    Parameters
    ----------
    name : str
        Unique configuration identifier. Must be a non-empty string.
    instructions : str or Path
        Plain text instructions or a path to a Jinja template file whose
        contents are loaded at runtime.
    tools : Sequence[object], optional
        Tool definitions associated with the configuration. Default is None.
    input_structure : Type[StructureBase], optional
        Structure class used to parse or validate input. Must subclass
        StructureBase. Default is None.
    output_structure : Type[StructureBase], optional
        Structure class used to format or validate output. Schema is
        automatically generated from this structure. Must subclass
        StructureBase. Default is None.
    system_vector_store : list[str], optional
        Optional list of vector store names to attach as system context.
        Default is None.
    add_output_instructions : bool, optional
        Whether to append output structure instructions to the prompt.
        Default is False.
    add_web_search_tool : bool, optional
        Whether to append a web_search tool to the tool list. Default is False.

    Raises
    ------
    TypeError
        If name is not a non-empty string.
        If instructions is not a string or Path.
        If tools is provided and is not a sequence.
        If input_structure or output_structure is not a class.
        If input_structure or output_structure does not subclass StructureBase.
    ValueError
        If instructions is a string that is empty or only whitespace.
    FileNotFoundError
        If instructions is a Path that does not point to a readable file.

    Methods
    -------
    __post_init__()
        Validate configuration invariants and enforce StructureBase subclassing.
    get_resolved_instructions()
        Return instructions with optional output structure guidance appended.
    get_resolved_tools()
        Return tools list with optional web_search tool appended.
    gen_response(openai_settings, data_path=None, tool_handlers=None)
        Build a ResponseBase instance from this configuration.
    to_json()
        Return a JSON-compatible dict representation (inherited from JSONSerializable).
    to_json_file(filepath)
        Write serialized JSON data to a file path (inherited from JSONSerializable).
    from_json(data)
        Create an instance from a JSON-compatible dict (class method, inherited from JSONSerializable).
    from_json_file(filepath)
        Load an instance from a JSON file (class method, inherited from JSONSerializable).

    Examples
    --------
    >>> configuration = ResponseConfiguration(
    ...     name="targeting_to_plan",
    ...     tools=None,
    ...     input_structure=PromptStructure,
    ...     output_structure=WebSearchStructure,
    ... )
    >>> configuration.name
    'prompt_to_websearch'
    """

    name: str
    instructions: str | Path
    tools: Optional[list]
    input_structure: Optional[Type[TIn]]
    output_structure: Optional[Type[TOut]]
    system_vector_store: Optional[list[str]] = None
    add_output_instructions: bool = False
    add_web_search_tool: bool = False

    def __post_init__(self) -> None:
        """Validate configuration invariants after initialization.

        Enforce non-empty naming, correct typing of structures, and ensure that
        any declared structure subclasses StructureBase.

        Raises
        ------
        TypeError
            If name is not a non-empty string.
            If tools is provided and is not a sequence.
            If input_structure or output_structure is not a class.
            If input_structure or output_structure does not subclass StructureBase.
        """
        if not self.name or not isinstance(self.name, str):
            raise TypeError("Configuration.name must be a non-empty str")

        instructions_value = self.instructions
        if isinstance(instructions_value, str):
            if not instructions_value.strip():
                raise ValueError("Configuration.instructions must be a non-empty str")
        elif isinstance(instructions_value, Path):
            instruction_path = instructions_value.expanduser()
            if not instruction_path.is_file():
                raise FileNotFoundError(
                    f"Instruction template not found: {instruction_path}"
                )
        else:
            raise TypeError("Configuration.instructions must be a str or Path")

        for attr in ("input_structure", "output_structure"):
            cls = getattr(self, attr)
            if cls is None:
                continue
            if not isinstance(cls, type):
                raise TypeError(
                    f"Configuration.{attr} must be a class (Type[StructureBase]) or None"
                )
            if not issubclass(cls, StructureBase):
                raise TypeError(f"Configuration.{attr} must subclass StructureBase")

        if self.tools is not None and not isinstance(self.tools, Sequence):
            raise TypeError("Configuration.tools must be a Sequence or None")

    @property
    def get_resolved_instructions(self) -> str:
        """Return the resolved instruction text.

        Returns
        -------
        str
            Plain-text instructions, loading template files when necessary.
        """
        resolved_instructions: str = resolve_instructions_from_path(self.instructions)
        output_instructions = ""
        if self.output_structure is not None and self.add_output_instructions:
            output_instructions = self.output_structure.get_prompt(
                add_enum_values=False
            )
            if output_instructions:
                return f"{resolved_instructions}\n{output_instructions}"

        return resolved_instructions

    @property
    def get_resolved_tools(self) -> list:
        """Return the complete list of tools, including optional web search tool.

        Returns
        -------
        list
            List of tool definitions, including web search tool if enabled.
        """
        tools = self.tools or []
        if self.add_web_search_tool:
            tools = tools + [{"type": "web_search"}]
        return tools

    def gen_response(
        self,
        *,
        openai_settings: OpenAISettings,
        data_path: Optional[Path] = None,
        tool_handlers: dict[str, ToolHandlerRegistration] | None = None,
    ) -> ResponseBase[TOut]:
        """Generate a ResponseBase instance based on the configuration.

        Parameters
        ----------
        openai_settings : OpenAISettings
            Authentication and model settings applied to the generated
            ResponseBase.
        data_path : Path or None, default None
            Optional override for the response artifact directory.
        tool_handlers : dict[str, ToolHandlerRegistration], optional
            Mapping of tool names to handler registrations. Registrations can include
            ToolSpec metadata to parse tool outputs by name. Defaults to an empty
            dictionary when not provided.

        Returns
        -------
        ResponseBase[TOut]
            An instance of ResponseBase configured with ``openai_settings``.
        """
        return ResponseBase[TOut](
            name=self.name,
            instructions=self.get_resolved_instructions,
            tools=self.get_resolved_tools,
            output_structure=self.output_structure,
            system_vector_store=self.system_vector_store,
            data_path=data_path,
            tool_handlers=tool_handlers,
            openai_settings=openai_settings,
        )


# Global default registry instance
_default_registry = ResponseRegistry()

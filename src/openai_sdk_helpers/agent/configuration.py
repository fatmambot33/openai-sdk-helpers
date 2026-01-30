"""Configuration helpers for ``AgentBase``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

from agents import Agent, Handoff, InputGuardrail, OutputGuardrail, Session
from agents.model_settings import ModelSettings

from ..utils.json.data_class import DataclassJSONSerializable
from ..utils.registry import RegistryBase
from ..utils.instructions import resolve_instructions_from_path
from ..structure.base import StructureBase
from ..settings import OpenAISettings


class AgentRegistry(RegistryBase["AgentConfiguration"]):
    """Registry for managing AgentConfiguration instances.

    Inherits from RegistryBase to provide centralized storage and retrieval
    of agent configurations, enabling reusable agent specs across the application.

    Examples
    --------
    >>> registry = AgentRegistry()
    >>> configuration = AgentConfiguration(
    ...     name="test_agent",
    ...     model="gpt-4o-mini",
    ...     instructions="Test instructions"
    ... )
    >>> registry.register(configuration)
    >>> retrieved = registry.get("test_agent")
    >>> retrieved.name
    'test_agent'
    """

    def load_from_directory(
        self,
        path: Path | str,
        *,
        config_class: type["AgentConfiguration"] | None = None,
    ) -> int:
        """Load all agent configurations from JSON files in a directory.

        Parameters
        ----------
        path : Path or str
            Directory path containing JSON configuration files.
        config_class : type[AgentConfiguration], optional
            The configuration class to use for deserialization.
            Defaults to AgentConfiguration.

        Returns
        -------
        int
            Number of configurations successfully loaded and registered.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        NotADirectoryError
            If the path is not a directory.

        Examples
        --------
        >>> registry = AgentRegistry()
        >>> count = registry.load_from_directory("./agents")
        >>> print(f"Loaded {count} configurations")
        """
        if config_class is None:
            config_class = AgentConfiguration
        return super().load_from_directory(path, config_class=config_class)


def get_default_registry() -> AgentRegistry:
    """Return the global default registry instance.

    Returns
    -------
    AgentRegistry
        Singleton registry for application-wide configuration storage.

    Examples
    --------
    >>> registry = get_default_registry()
    >>> configuration = AgentConfiguration(
    ...     name="test", model="gpt-4o-mini", instructions="Test instructions"
    ... )
    >>> registry.register(configuration)
    """
    return _default_registry


@dataclass(frozen=True, slots=True)
class AgentConfiguration(DataclassJSONSerializable):
    """Immutable configuration for building a AgentBase.

    Encapsulates all metadata required to define an agent including its
    instructions, tools, model settings, handoffs, guardrails, and session
    management. Inherits from DataclassJSONSerializable to support serialization.

    This dataclass is frozen (immutable) to ensure thread-safety and
    enable use as dictionary keys. All list-type fields use None as the
    default value rather than mutable defaults like [] to avoid issues
    with shared state across instances.

    Parameters
    ----------
    name : str
        Unique identifier for the agent. Must be a non-empty string.
    instructions : str or Path
        Plain text instructions or a path to a Jinja template file whose
        contents are loaded at runtime. Required field.
    description : str, optional
        Short description of the agent's purpose. Default is None.
    model : str, optional
        Model identifier to use (e.g., "gpt-4o-mini"). Default is None.
    template_path : str or Path, optional
        Path to the Jinja template (absolute or relative to prompt_dir).
        This takes precedence over instructions if both are provided.
        Default is None.
    input_structure : type[StructureBase], optional
        Structure class describing the agent input. Default is None.
    output_structure : type[StructureBase], optional
        Structure class describing the agent output. Default is None.
    tools : list, optional
        Tool definitions available to the agent. Default is None.
    model_settings : ModelSettings, optional
        Additional model configuration settings. Default is None.
    handoffs : list[Agent or Handoff], optional
        List of agents or handoff configurations that this agent can
        delegate to for specific tasks. Default is None.
    input_guardrails : list[InputGuardrail], optional
        List of guardrails to validate agent inputs before processing.
        Default is None.
    output_guardrails : list[OutputGuardrail], optional
        List of guardrails to validate agent outputs before returning.
        Default is None.
    session : Session, optional
        Session configuration for automatically maintaining conversation
        history across agent runs. Default is None.

    Methods
    -------
    __post_init__()
        Validate configuration invariants after initialization.
    instructions_text
        Return the resolved instruction content as a string.
    resolve_prompt_path(prompt_dir)
        Resolve the prompt template path for this configuration.
    gen_agent(run_context_wrapper)
        Create a AgentBase instance from this configuration.
    to_openai_settings(dotenv_path=None, **overrides)
        Build OpenAISettings using this configuration as defaults.
    replace(**changes)
        Create a new AgentConfiguration with specified fields replaced.
    to_json()
        Return a JSON-compatible dict (inherited from JSONSerializable).
    to_json_file(filepath)
        Write serialized JSON data to a file (inherited from JSONSerializable).
    from_json(data)
        Create an instance from a JSON-compatible dict (inherited from JSONSerializable).
    from_json_file(filepath)
        Load an instance from a JSON file (inherited from JSONSerializable).

    Examples
    --------
    >>> configuration = AgentConfiguration(
    ...     name="summarizer",
    ...     description="Summarizes text",
    ...     model="gpt-4o-mini"
    ... )
    >>> configuration.name
    'summarizer'
    """

    name: str
    instructions: str | Path
    description: str | None = None
    model: str | None = None
    template_path: str | Path | None = None
    input_structure: type[StructureBase] | None = None
    output_structure: type[StructureBase] | None = None
    tools: list | None = None
    model_settings: ModelSettings | None = None
    handoffs: list[Agent | Handoff] | None = None
    input_guardrails: list[InputGuardrail] | None = None
    output_guardrails: list[OutputGuardrail] | None = None
    session: Session | None = None
    add_output_instructions: bool = False
    add_web_search_tool: bool = False

    def __post_init__(self) -> None:
        """Validate configuration invariants after initialization.

        Ensures that the name is a non-empty string and that instructions
        are properly formatted.

        Raises
        ------
        TypeError
            If name is not a non-empty string.
            If instructions is not a string or Path.
            If input_structure or output_structure is not a class.
            If input_structure or output_structure does not subclass StructureBase.
        ValueError
            If instructions is an empty string.
        FileNotFoundError
            If instructions is a Path that doesn't point to a readable file.
        """
        if not self.name or not isinstance(self.name, str):
            raise TypeError("AgentConfiguration.name must be a non-empty str")

        # Validate instructions (required field, like in Response module)
        instructions_value = self.instructions
        if isinstance(instructions_value, str):
            if not instructions_value.strip():
                raise ValueError(
                    "AgentConfiguration.instructions must be a non-empty str"
                )
        elif isinstance(instructions_value, Path):
            instruction_path = instructions_value.expanduser()
            if not instruction_path.is_file():
                raise FileNotFoundError(
                    f"Instruction template not found: {instruction_path}"
                )
        else:
            raise TypeError("AgentConfiguration.instructions must be a str or Path")

        for attr in ("input_structure", "output_structure"):
            cls = getattr(self, attr)
            if cls is None:
                continue
            if not isinstance(cls, type):
                raise TypeError(
                    f"AgentConfiguration.{attr} must be a class (Type[StructureBase]) or None"
                )
            if not issubclass(cls, StructureBase):
                raise TypeError(
                    f"AgentConfiguration.{attr} must subclass StructureBase"
                )

        if self.template_path is not None and isinstance(self.template_path, Path):
            # Validate template_path if it's a Path object
            template = self.template_path.expanduser()
            if not template.exists():
                # We don't raise here because template_path might be relative
                # and resolved later with prompt_dir
                pass

    @property
    def instructions_text(self) -> str:
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

    def _resolve_instructions(self) -> str:
        """Resolve instructions from string or file path."""
        return resolve_instructions_from_path(self.instructions)

    def to_openai_settings(
        self, *, dotenv_path: Path | None = None, **overrides: Any
    ) -> OpenAISettings:
        """Build OpenAI settings using this configuration as defaults.

        Parameters
        ----------
        dotenv_path : Path or None, optional
            Optional dotenv file path for loading environment variables.
        overrides : Any
            Keyword overrides applied on top of environment values. Use this
            to supply API credentials and override defaults.

        Returns
        -------
        OpenAISettings
            OpenAI settings instance with defaults derived from this
            configuration.

        Raises
        ------
        ValueError
            If no API key is supplied via overrides or environment variables.

        Examples
        --------
        >>> configuration = AgentConfiguration(
        ...     name="summarizer",
        ...     instructions="Summarize text",
        ...     model="gpt-4o-mini",
        ... )
        >>> settings = configuration.to_openai_settings(api_key="sk-...")
        >>> # Or rely on environment variables like OPENAI_API_KEY
        >>> settings = configuration.to_openai_settings()
        """
        if self.model and "default_model" not in overrides:
            overrides["default_model"] = self.model
        return OpenAISettings.from_env(dotenv_path=dotenv_path, **overrides)

    def resolve_prompt_path(self, prompt_dir: Path | None = None) -> Path | None:
        """Resolve the prompt template path for this configuration.

        Parameters
        ----------
        prompt_dir : Path or None, default=None
            Directory holding prompt templates when a relative path is needed.

        Returns
        -------
        Path or None
            Resolved prompt path if a template is configured or discovered.
        """
        if self.template_path:
            return Path(self.template_path)
        if prompt_dir is not None:
            return prompt_dir / f"{self.name}.jinja"
        return None

    def gen_agent(
        self,
        run_context_wrapper: Any = None,
    ) -> Any:
        """Create a AgentBase instance from this configuration.

        This is a convenience method that instantiates ``AgentBase`` directly.

        Parameters
        ----------
        run_context_wrapper : RunContextWrapper or None, default=None
            Optional wrapper providing runtime context for prompt rendering.

        Returns
        -------
        AgentBase
            Configured agent instance ready for execution.

        Examples
        --------
        >>> configuration = AgentConfiguration(
        ...     name="helper", model="gpt-4o-mini", instructions="Help the user"
        ... )
        >>> agent = configuration.gen_agent()
        >>> result = agent.run_sync("Hello!")
        """
        # Import here to avoid circular dependency
        from .base import AgentBase

        return AgentBase(
            configuration=self,
            run_context_wrapper=run_context_wrapper,
        )

    def replace(self, **changes: Any) -> AgentConfiguration:
        """Create a new AgentConfiguration with specified fields replaced.

        Since AgentConfiguration is frozen (immutable), this method creates a new
        instance with the specified changes applied. This is useful for
        creating variations of a configuration.

        Parameters
        ----------
        **changes : Any
            Keyword arguments specifying fields to change and their new values.

        Returns
        -------
        AgentConfiguration
            New configuration instance with changes applied.

        Examples
        --------
        >>> configuration = AgentConfiguration(
        ...     name="agent1", model="gpt-4o-mini", instructions="Agent instructions"
        ... )
        >>> config2 = configuration.replace(name="agent2", description="Modified")
        >>> config2.name
        'agent2'
        >>> config2.model
        'gpt-4o-mini'
        """
        from dataclasses import replace

        return replace(self, **changes)

    def to_response_config(self) -> Any:
        """Convert this AgentConfiguration to a ResponseConfiguration.

        This is a convenience method for creating a ResponseConfiguration
        instance using the relevant fields from this agent configuration.

        Returns
        -------
        ResponseConfiguration
            New response configuration instance.

        Examples
        --------
        >>> agent_config = AgentConfiguration(
        ...     name="responder",
        ...     model="gpt-4o-mini",
        ...     instructions="Respond to user queries"
        ... )
        >>> response_config = agent_config.to_response_config()
        >>> response_config.name
        'responder'
        """
        from ..response.configuration import ResponseConfiguration

        return ResponseConfiguration(
            name=self.name,
            instructions=self.instructions,
            input_structure=self.input_structure,
            output_structure=self.output_structure,
            tools=self.tools,
        )


# Global default registry instance
_default_registry = AgentRegistry()

__all__ = ["AgentConfiguration", "AgentRegistry", "get_default_registry"]

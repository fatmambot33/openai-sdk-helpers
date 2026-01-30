"""Base agent helpers built on the OpenAI Agents SDK."""

from __future__ import annotations

import logging
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Protocol, cast

from agents import Agent, Handoff, InputGuardrail, OutputGuardrail, Session
from agents.model_settings import ModelSettings
from agents.run_context import RunContextWrapper
from agents.tool import Tool
from jinja2 import Template

from ..environment import get_data_path
from ..utils.json.data_class import DataclassJSONSerializable
from ..structure.base import StructureBase
from ..tools import (
    StructureType,
    ToolHandlerRegistration,
    ToolSpec,
)

from ..utils import (
    check_filepath,
    log,
)

from .runner import run_async, run_sync

if TYPE_CHECKING:
    from ..settings import OpenAISettings
    from ..response.base import ResponseBase
    from ..files_api import FilePurpose, FilesAPIManager


class AgentConfigurationProtocol(Protocol):
    """Protocol describing the configuration attributes for AgentBase."""

    @property
    def name(self) -> str:
        """Agent name."""
        ...

    @property
    def description(self) -> Optional[str]:
        """Agent description."""
        ...

    @property
    def template_path(self) -> Optional[str | Path]:
        """Template path."""
        ...

    @property
    def model(self) -> Optional[str]:
        """Model identifier."""
        ...

    @property
    def instructions(self) -> str | Path:
        """Instructions."""
        ...

    @property
    def instructions_text(self) -> str:
        """Resolved instructions text."""
        ...

    @property
    def input_structure(self) -> Optional[type[StructureBase]]:
        """Input type."""
        ...

    @property
    def output_structure(self) -> Optional[type[StructureBase]]:
        """Output type."""
        ...

    @property
    def tools(self) -> Optional[list]:
        """Tools."""
        ...

    @property
    def model_settings(self) -> Optional[ModelSettings]:
        """Model settings."""
        ...

    @property
    def handoffs(self) -> Optional[list[Agent | Handoff]]:
        """Handoffs."""
        ...

    @property
    def input_guardrails(self) -> Optional[list[InputGuardrail]]:
        """Input guardrails."""
        ...

    @property
    def output_guardrails(self) -> Optional[list[OutputGuardrail]]:
        """Output guardrails."""
        ...

    @property
    def session(self) -> Optional[Session]:
        """Session."""
        ...


class AgentBase(DataclassJSONSerializable):
    """Factory for creating and configuring specialized agents.

    ``AgentBase`` provides the foundation for building OpenAI agents with support
    for Jinja2 prompt templates, custom tools, handoffs for agent delegation,
    input and output guardrails for validation, session management for
    conversation history, and both synchronous and asynchronous execution modes.
    All specialized agents in this package extend this base class.

    Examples
    --------
    Create a basic agent from configuration:

    >>> from openai_sdk_helpers.agent import AgentBase, AgentConfiguration
    >>> configuration = AgentConfiguration(
    ...     name="my_agent",
    ...     description="A custom agent",
    ...     model="gpt-4o-mini"
    ... )
    >>> agent = AgentBase(configuration=configuration)
    >>> result = agent.run_sync("What is 2+2?")

    Use absolute path to template:

    >>> configuration = AgentConfiguration(
    ...     name="my_agent",
    ...     template_path="/absolute/path/to/template.jinja",
    ...     model="gpt-4o-mini"
    ... )
    >>> agent = AgentBase(configuration=configuration)

    Use async execution:

    >>> import asyncio
    >>> async def main():
    ...     result = await agent.run_async("Explain quantum physics")
    ...     return result
    >>> asyncio.run(main())

    Methods
    -------
    build_prompt_from_jinja(run_context_wrapper)
        Render the agent prompt using Jinja and optional context.
    get_prompt(run_context_wrapper, _)
        Render the agent prompt using the provided run context.
    name
        Return the name of this agent.
    instructions_text
        Return the resolved instructions for this agent.
    tools
        Return the tools configured for this agent.
    output_structure
        Return the output type configured for this agent.
    model_settings
        Return the model settings configured for this agent.
    handoffs
        Return the handoff configurations for this agent.
    input_guardrails
        Return the input guardrails configured for this agent.
    output_guardrails
        Return the output guardrails configured for this agent.
    session
        Return the session configured for this agent.
    get_agent()
        Construct the configured :class:`agents.Agent` instance.
    run_async(input, context, output_structure, session)
        Execute the agent asynchronously and optionally cast the result.
    run_sync(input, context, output_structure, session)
        Execute the agent synchronously.
    as_tool()
        Return the agent as a callable tool.
    as_response_tool()
        Return response tool handler and definition for Responses API use.
    build_response(openai_settings, data_path=None, tool_handlers=None, system_vector_store=None)
        Build a ResponseBase instance based on this agent.
    build_input_messages(content, files=None, files_manager=None, file_purpose="user_data", image_detail="auto")
        Build Agents SDK input messages with optional file attachments.
    save_error(exc)
        Persist error details to a file named with the agent UUID.
    close()
        Clean up agent resources (can be overridden by subclasses).
    """

    def __init__(
        self,
        *,
        configuration: AgentConfigurationProtocol,
        run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None,
        data_path: Path | str | None = None,
    ) -> None:
        """Initialize the AgentBase using a configuration object.

        Parameters
        ----------
        configuration : AgentConfigurationProtocol
            Configuration describing this agent.
        run_context_wrapper : RunContextWrapper or None, default=None
            Optional wrapper providing runtime context for prompt rendering.
        data_path : Path | str | None, default=None
            Optional base path for storing agent data.
        """
        self._configuration = configuration
        self.uuid = uuid.uuid4()
        self._model = configuration.model
        if self._model is None:
            raise ValueError(
                f"Model must be specified in configuration for agent '{configuration.name}'."
            )

        # Build template from file or fall back to instructions
        self._template_path = configuration.template_path
        if self._template_path is None:
            instructions_text = configuration.instructions_text
            self._template = Template(instructions_text)
            self._instructions = instructions_text
        else:
            self._template_path = Path(self._template_path)
            if not self._template_path.exists():
                raise FileNotFoundError(
                    f"Template for agent '{self._configuration.name}' not found at {self._template_path}."
                )
            self._template = Template(self._template_path.read_text(encoding="utf-8"))
            self._instructions = None

        # Resolve data_path with class name appended
        class_name = self.__class__.__name__
        if data_path is not None:
            data_path_obj = Path(data_path)
            if data_path_obj.name == class_name:
                self._data_path = data_path_obj
            else:
                self._data_path = data_path_obj / class_name
        else:
            self._data_path = get_data_path(self.__class__.__name__)

        self._input_structure = configuration.input_structure
        self._output_structure = (
            configuration.output_structure or configuration.input_structure
        )
        self._tools = configuration.tools
        self._model_settings = configuration.model_settings
        self._handoffs = configuration.handoffs
        self._input_guardrails = configuration.input_guardrails
        self._output_guardrails = configuration.output_guardrails
        self._session = configuration.session
        self._run_context_wrapper = run_context_wrapper

    def _build_prompt_from_jinja(self) -> str:
        """Render the instructions prompt for this agent.

        Returns
        -------
        str
            Prompt text rendered from the Jinja template.
        """
        return self.build_prompt_from_jinja(
            run_context_wrapper=self._run_context_wrapper
        )

    def build_prompt_from_jinja(
        self, run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None
    ) -> str:
        """Render the agent prompt using the provided run context.

        Parameters
        ----------
        run_context_wrapper : RunContextWrapper or None, default=None
            Wrapper whose ``context`` dictionary is used to render the Jinja
            template.

        Returns
        -------
        str
            Rendered prompt text.
        """
        context = {}
        if run_context_wrapper is not None:
            context = run_context_wrapper.context

        return self._template.render(context)

    def get_prompt(
        self, run_context_wrapper: RunContextWrapper[Dict[str, Any]], *, _: Agent
    ) -> str:
        """Render the agent prompt using the provided run context.

        Parameters
        ----------
        run_context_wrapper : RunContextWrapper
            Wrapper around the current run context whose ``context`` dictionary
            is used to render the Jinja template.
        _ : Agent
            Underlying Agent instance (ignored).

        Returns
        -------
        str
            The rendered prompt.
        """
        return self.build_prompt_from_jinja(run_context_wrapper)

    @property
    def name(self) -> str:
        """Return the name of this agent.

        Returns
        -------
        str
            Name used to identify the agent.
        """
        return self._configuration.name

    @property
    def description(self) -> Optional[str]:
        """Return the description of this agent.

        Returns
        -------
        str or None
            Description of the agent's purpose.
        """
        return self._configuration.description

    @property
    def model(self) -> str:
        """Return the model identifier for this agent.

        Returns
        -------
        str
            Model identifier used by the agent.
        """
        return self._model  # pyright: ignore[reportReturnType]

    @property
    def instructions_text(self) -> str:
        """Return the resolved instructions for this agent.

        Returns
        -------
        str
            Rendered instructions text using the current run context.
        """
        return self._configuration.instructions_text

    @property
    def tools(self) -> Optional[list]:
        """Return the tools configured for this agent.

        Returns
        -------
        list or None
            Tool definitions configured for the agent.
        """
        return self._configuration.tools

    @property
    def output_structure(self) -> Optional[type[StructureBase]]:
        """Return the output type configured for this agent.

        Returns
        -------
        type[StructureBase] or None
            Output type used to cast responses.
        """
        return self._configuration.output_structure

    @property
    def model_settings(self) -> Optional[ModelSettings]:
        """Return the model settings configured for this agent.

        Returns
        -------
        ModelSettings or None
            Model settings applied to the agent.
        """
        return self._model_settings

    @property
    def handoffs(self) -> Optional[list[Agent | Handoff]]:
        """Return the handoff configurations for this agent.

        Returns
        -------
        list[Agent or Handoff] or None
            Handoff configurations available to the agent.
        """
        return self._handoffs

    @property
    def input_guardrails(self) -> Optional[list[InputGuardrail]]:
        """Return the input guardrails configured for this agent.

        Returns
        -------
        list[InputGuardrail] or None
            Input guardrails applied to the agent.
        """
        return self._input_guardrails

    @property
    def output_guardrails(self) -> Optional[list[OutputGuardrail]]:
        """Return the output guardrails configured for this agent.

        Returns
        -------
        list[OutputGuardrail] or None
            Output guardrails applied to the agent.
        """
        return self._output_guardrails

    @property
    def session(self) -> Optional[Session]:
        """Return the session configured for this agent.

        Returns
        -------
        Session or None
            Session configuration used for maintaining conversation history.
        """
        return self._session

    def get_agent(
        self, output_structure: Optional[type[StructureBase]] = None
    ) -> Agent:
        """Construct and return the configured :class:`agents.Agent` instance.

        Parameters
        ----------
        output_structure : type[StructureBase] or None, default=None
            Optional override for the agent output schema.

        Returns
        -------
        Agent
            Initialized agent ready for execution.
        """
        agent_config: Dict[str, Any] = {
            "name": self._configuration.name,
            "instructions": self._configuration.instructions_text or ".",
            "model": self._model,
        }
        output_type = output_structure or self._configuration.output_structure
        if output_type is not None:
            agent_config["output_type"] = output_type
        if self._configuration.tools:
            agent_config["tools"] = self._configuration.tools
        if self._model_settings:
            agent_config["model_settings"] = self._model_settings
        if self._handoffs:
            agent_config["handoffs"] = self._handoffs
        if self._input_guardrails:
            agent_config["input_guardrails"] = self._input_guardrails
        if self._output_guardrails:
            agent_config["output_guardrails"] = self._output_guardrails

        return Agent(**agent_config)

    async def run_async(
        self,
        input: str | list[dict[str, Any]],
        *,
        context: Optional[Dict[str, Any]] = None,
        output_structure: Optional[type[StructureBase]] = None,
        session: Optional[Any] = None,
    ) -> Any:
        """Execute the agent asynchronously.

        Parameters
        ----------
        input : str or list[dict[str, Any]]
            Prompt text or structured input for the agent.
        context : dict or None, default=None
            Optional dictionary passed to the agent.
        output_structure : type[StructureBase] or None, default=None
            Optional type used to cast the final output.
        session : Session or None, default=None
            Optional session for maintaining conversation history across runs.
            If not provided, uses the session from configuration if available.

        Returns
        -------
        Any
            Agent result, optionally converted to ``output_structure``.
        """
        if self._output_structure is not None and output_structure is None:
            output_structure = self._output_structure
        # Use session from parameter, fall back to configuration session
        session_to_use = session if session is not None else self._session
        try:
            return await run_async(
                agent=self.get_agent(output_structure=output_structure),
                input=input,
                context=context,
                output_structure=output_structure,
                session=session_to_use,
            )
        except Exception as exc:
            try:
                self.save_error(exc)
            except Exception as save_exc:
                log(
                    f"Failed to save error details for agent {self.uuid}: {save_exc}",
                    level=logging.ERROR,
                    exc=save_exc,
                )
            log(
                f"Error running agent '{self.name}': {exc}",
                level=logging.ERROR,
                exc=exc,
            )
            raise

    def run_sync(
        self,
        input: str | list[dict[str, Any]],
        *,
        context: Optional[Dict[str, Any]] = None,
        output_structure: Optional[type[StructureBase]] = None,
        session: Optional[Any] = None,
    ) -> Any:
        """Run the agent synchronously.

        Parameters
        ----------
        input : str or list[dict[str, Any]]
            Prompt text or structured input for the agent.
        context : dict or None, default=None
            Optional dictionary passed to the agent.
        output_structure : type[StructureBase] or None, default=None
            Optional type used to cast the final output.
        session : Session or None, default=None
            Optional session for maintaining conversation history across runs.
            If not provided, uses the session from configuration if available.

        Returns
        -------
        Any
            Agent result, optionally converted to ``output_structure``.
        """
        if self._output_structure is not None and output_structure is None:
            output_structure = self._output_structure
        # Use session from parameter, fall back to configuration session
        session_to_use = session if session is not None else self._session
        try:
            return run_sync(
                agent=self.get_agent(output_structure=output_structure),
                input=input,
                context=context,
                output_structure=output_structure,
                session=session_to_use,
            )
        except Exception as exc:
            try:
                self.save_error(exc)
            except Exception as save_exc:
                log(
                    f"Failed to save error details for agent {self.uuid}: {save_exc}",
                    level=logging.ERROR,
                    exc=save_exc,
                )
            log(
                f"Error running agent '{self.name}': {exc}",
                level=logging.ERROR,
                exc=exc,
            )
            raise

    def as_tool(self) -> Tool:
        """Return the agent as a callable tool.

        Returns
        -------
        Tool
            Tool instance wrapping this agent.
        """
        agent = self.get_agent()
        tool_obj: Tool = agent.as_tool(
            tool_name=self._configuration.name,
            tool_description=self._configuration.description,
        )
        return tool_obj

    def as_tool_handler_registration(
        self,
    ) -> ToolHandlerRegistration:
        """Return the agent as a ToolHandlerRegistration for Responses API use.

        Parameters
        ----------
        tool_name : str or None, default=None
            Optional override for the tool name. When None, uses the agent name.
        """
        tool_spec = ToolSpec(
            tool_name=self.name,
            tool_description=self.description,
            input_structure=cast(StructureType, self._configuration.input_structure),
            output_structure=cast(StructureType, self._configuration.output_structure),
        )
        return ToolHandlerRegistration(handler=self.run_sync, tool_spec=tool_spec)

    def build_response(
        self,
        *,
        openai_settings: OpenAISettings,
        data_path: Path | str | None = None,
        tool_handlers: dict[str, ToolHandlerRegistration] | None = None,
        system_vector_store: list[str] | None = None,
    ) -> ResponseBase[StructureBase]:
        """Build a ResponseBase instance from this agent configuration.

        Parameters
        ----------
        openai_settings : OpenAISettings
            Authentication and model settings applied to the generated response.
        data_path : Path, str, or None, default None
            Optional path for storing response artifacts. When None, the
            response uses the default data directory.
        tool_handlers : dict[str, ToolHandlerRegistration] or None, default None
            Optional mapping of tool names to handler registrations. Registrations
            can include ToolSpec metadata to parse tool outputs by name.
        system_vector_store : list[str] or None, default None
            Optional list of vector store names to attach as system context.

        Returns
        -------
        ResponseBase[StructureBase]
            ResponseBase instance configured with this agent's settings.

        Examples
        --------
        >>> from openai_sdk_helpers import OpenAISettings
        >>> response = agent.build_response(openai_settings=OpenAISettings.from_env())
        """
        from ..response.base import ResponseBase
        from ..settings import OpenAISettings

        if not isinstance(openai_settings, OpenAISettings):
            raise TypeError("openai_settings must be an OpenAISettings instance")

        tools = self._normalize_response_tools(self.tools)

        return ResponseBase(
            name=self._configuration.name,
            instructions=self._configuration.instructions_text,
            tools=tools,
            output_structure=self.output_structure,
            system_vector_store=system_vector_store,
            data_path=data_path,
            tool_handlers=tool_handlers,
            openai_settings=openai_settings,
        )

    @staticmethod
    def build_input_messages(
        content: str | list[str],
        files: str | list[str] | None = None,
        *,
        files_manager: FilesAPIManager | None = None,
        openai_settings: OpenAISettings | None = None,
        file_purpose: FilePurpose = "user_data",
        image_detail: Literal["low", "high", "auto"] = "auto",
    ) -> list[dict[str, Any]]:
        """Build Agents SDK input messages with file attachments.

        Parameters
        ----------
        content : str or list[str]
            Prompt text or list of prompt texts to send.
        files : str, list[str], or None, default None
            Optional file path or list of file paths. Image files are sent as
            base64-encoded ``input_image`` entries. Document files are uploaded
            using ``files_manager`` and sent as ``input_file`` entries.
        files_manager : FilesAPIManager or None, default None
            File upload helper used to create file IDs for document uploads.
            Required when ``files`` contains non-image documents.
        openai_settings : OpenAISettings or None, default None
            Optional OpenAI settings used to build a FilesAPIManager when one is
            not provided. When supplied, ``openai_settings.create_client()`` is
            used to initialize the Files API manager.
        file_purpose : FilePurpose, default "user_data"
            Purpose passed to the Files API when uploading document files.
        image_detail : {"low", "high", "auto"}, default "auto"
            Detail hint passed along with base64-encoded image inputs.

        Returns
        -------
        list[dict[str, Any]]
            Agents SDK input messages that include text and optional file entries.

        Raises
        ------
        ValueError
            If document files are provided without a ``files_manager``.

        Examples
        --------
        >>> from openai import OpenAI
        >>> from openai_sdk_helpers.files_api import FilesAPIManager
        >>> client = OpenAI()
        >>> files_manager = FilesAPIManager(client)
        >>> messages = AgentBase.build_input_messages(
        ...     "Summarize this document",
        ...     files="report.pdf",
        ...     files_manager=files_manager,
        ... )
        """
        from .files import build_agent_input_messages

        return build_agent_input_messages(
            content=content,
            files=files,
            files_manager=files_manager,
            openai_settings=openai_settings,
            file_purpose=file_purpose,
            image_detail=image_detail,
        )

    def _build_response_parameters(self) -> dict[str, Any]:
        """Build the Responses API parameter schema for this agent tool.

        Returns
        -------
        dict[str, Any]
            JSON schema describing tool input parameters.
        """
        if self._input_structure is not None:
            return self._input_structure.get_schema()
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt text to run."}
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    @staticmethod
    def _normalize_response_tools(tools: Optional[list]) -> Optional[list]:
        """Normalize tool definitions for the Responses API."""
        if not tools:
            return tools

        normalized: list[Any] = []
        for tool in tools:
            if hasattr(tool, "to_dict") and callable(tool.to_dict):
                normalized.append(tool.to_dict())
            elif hasattr(tool, "to_openai_tool") and callable(tool.to_openai_tool):
                normalized.append(tool.to_openai_tool())
            elif hasattr(tool, "schema"):
                normalized.append(tool.schema)
            else:
                normalized.append(tool)
        return normalized

    def __enter__(self) -> AgentBase:
        """Enter the context manager for resource management.

        Returns
        -------
        AgentBase
            Self reference for use in with statements.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and clean up resources.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred, otherwise None.
        exc_val : Exception or None
            Exception instance if an exception occurred, otherwise None.
        exc_tb : traceback or None
            Traceback object if an exception occurred, otherwise None.
        """
        self.close()

    def close(self) -> None:
        """Clean up agent resources.

        This method is called automatically when using the agent as a
        context manager. Override in subclasses to implement custom
        cleanup logic.

        Examples
        --------
        >>> agent = AgentBase(configuration)
        >>> try:
        ...     result = agent.run_sync("query")
        ... finally:
        ...     agent.close()
        """
        log(f"Closing session {self.uuid} for {self.__class__.__name__}")
        self.save()

    def __repr__(self) -> str:
        """Return a string representation of the AgentBase.

        Returns
        -------
        str
            String representation including agent name and model.
        """
        return f"<AgentBase name={self.name!r} model={self.model!r}>"

    def save(self, filepath: str | Path | None = None) -> None:
        """Serialize the message history to a JSON file.

        Saves the current message history to a specified file path in JSON format.
        If no file path is provided, it saves to a default location based on
        the agent's UUID.

        Parameters
        ----------
        filepath : str | Path | None, default=None
            Optional file path to save the serialized history. If None,
            uses a default filename based on the agent name.
        """
        if filepath is not None:
            target = Path(filepath)
        else:
            filename = f"{str(self.uuid).lower()}.json"
            target = self._data_path / self.name / filename

        checked = check_filepath(filepath=target)
        self.to_json_file(filepath=checked)
        log(f"Saved messages to {target}")

    def save_error(self, exc: BaseException) -> Path:
        """Persist error details to a file named with the agent UUID.

        Parameters
        ----------
        exc : BaseException
            Exception instance to serialize.

        Returns
        -------
        Path
            Path to the error file written to disk.

        Examples
        --------
        >>> try:
        ...     agent.run_sync("trigger error")
        ... except Exception as exc:
        ...     agent.save_error(exc)
        """
        error_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        filename = f"{str(self.uuid).lower()}_error.txt"
        target = self._data_path / self.name / filename
        checked = check_filepath(filepath=target)
        checked.write_text(error_text, encoding="utf-8")
        log(f"Saved error details to {checked}")
        return checked


__all__ = ["AgentConfigurationProtocol", "AgentBase"]

"""Core response management for OpenAI API interactions.

This module implements the ResponseBase class, which manages the complete
lifecycle of OpenAI API interactions including input construction, tool
execution, message history, vector store attachments, and structured output
parsing.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import traceback
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Sequence,
    TypeVar,
    cast,
)

from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)
from openai.types.responses.response_input_file_content_param import (
    ResponseInputFileContentParam,
)
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_content_param import (
    ResponseInputImageContentParam,
)
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_param import ResponseInputItemParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_message import ResponseOutputMessage

from .messages import ResponseMessage, ResponseMessages
from ..settings import OpenAISettings
from ..structure import StructureBase
from ..types import OpenAIClient
from ..tools import ToolHandlerRegistration, ToolSpec
from ..utils import (
    check_filepath,
    coerce_jsonable,
    customJSONEncoder,
    ensure_list,
    log,
)

if TYPE_CHECKING:  # pragma: no cover - only for typing hints
    from openai_sdk_helpers.streamlit_app.configuration import StreamlitAppConfig

T = TypeVar("T", bound=StructureBase)
RB = TypeVar("RB", bound="ResponseBase[StructureBase]")


class ResponseBase(Generic[T]):
    """Manage OpenAI API interactions for structured responses.

    Orchestrates the complete lifecycle of OpenAI API requests including
    input construction, tool execution, message history management, vector
    store attachments, and structured output parsing. Supports both
    synchronous and asynchronous execution with automatic resource cleanup.

    The class handles conversation state, tool calls with custom handlers,
    file attachments via vector stores, and optional parsing into typed
    structured output models. Sessions can be persisted to disk and restored.

    Parameters
    ----------
    name : str
        Name for this response session, used for organizing artifacts
        and naming vector stores.
    instructions : str
        System instructions provided to the OpenAI API for context.
    tools : list[ToolHandlerRegistration] or None
        Tool handler registrations for the OpenAI API request. Pass None for
        no tools.
    output_structure : type[StructureBase] or None
        Structure class used to parse tool call outputs. When provided,
        the schema is automatically generated using the structure's
        response_format() method. Pass None for unstructured responses.
    system_vector_store : list[str] or None, default None
        Optional list of vector store names to attach as system context.
    data_path : Path, str, or None, default None
        Optional absolute directory path for storing artifacts. If not provided,
        defaults to get_data_path(class_name). Session files are saved as
        data_path / uuid.json.
    tool_handlers : dict[str, ToolHandlerRegistration] or None, default None
        Mapping from tool names to handler registrations that include optional
        ToolSpec metadata to parse tool outputs by name. Each handler receives
        a ResponseFunctionToolCall and returns a string or any serializable
        result. Defaults to an empty dict when not provided.
    openai_settings : OpenAISettings or None, default None
        Fully configured OpenAI settings with API key and default model.
        Required for normal operation.

    Attributes
    ----------
    uuid : UUID
        Unique identifier for this response session.
    name : str
        Lowercase class name used for path construction.
    instructions_text : str
        System instructions provided to the OpenAI API for context.
    messages : ResponseMessages
        Complete message history for this session.
    output_structure : type[T] | None
        Structure class used to parse tool call outputs, or None.
    tools : list | None
        Tool definitions provided to the OpenAI API, or None.


    Methods
    -------
    run_async(content, files=None, use_vector_store=False)
        Generate a response asynchronously and return parsed output.
    run_sync(content, files=None, use_vector_store=False)
        Execute run_async synchronously with thread management.
    register_tool(func, tool_spec)
        Register a tool handler and definition from a ToolSpec.
    get_last_tool_message()
        Return the most recent tool message or None.
    get_last_user_message()
        Return the most recent user message or None.
    get_last_assistant_message()
        Return the most recent assistant message or None.
    build_streamlit_config(**kwargs)
        Construct a StreamlitAppConfig using this class as the builder.
    save(filepath=None)
        Serialize the message history to a JSON file.
    save_error(exc)
        Persist error details to a file named with the response UUID.
    close()
        Clean up remote resources including vector stores.

    Examples
    --------
    >>> from openai_sdk_helpers import ResponseBase, OpenAISettings
    >>> settings = OpenAISettings(api_key="...", default_model="gpt-4")
    >>> response = ResponseBase(
    ...     instructions="You are a helpful assistant",
    ...     tools=None,
    ...     output_structure=None,
    ...     tool_handlers={},
    ...     openai_settings=settings
    ... )
    >>> result = response.run_sync("Hello, world!")
    >>> response.close()
    """

    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        tools: list[ToolHandlerRegistration] | None,
        output_structure: type[T] | None,
        system_vector_store: list[str] | None = None,
        data_path: Path | str | None = None,
        tool_handlers: dict[str, ToolHandlerRegistration] | None = None,
        openai_settings: OpenAISettings | None = None,
    ) -> None:
        """Initialize a response session with OpenAI configuration.

        Sets up the OpenAI client, message history, vector stores, and tool
        handlers for a complete response workflow. The session can optionally
        be persisted to disk for later restoration.

        Parameters
        ----------
        name : str
            Name for this response session, used for organizing artifacts
            and naming vector stores.
        instructions : str
            System instructions provided to the OpenAI API for context.
        tools : list[ToolHandlerRegistration] or None
            Tool handler registrations for the OpenAI API request. Pass None for
            no tools.
        output_structure : type[StructureBase] or None
            Structure class used to parse tool call outputs. When provided,
            the schema is automatically generated using the structure's
            response_format() method. Pass None for unstructured responses.
        system_vector_store : list[str] or None, default None
            Optional list of vector store names to attach as system context.
        data_path : Path, str, or None, default None
            Optional absolute directory path for storing artifacts. If not provided,
            defaults to get_data_path(class_name). Session files are saved as
            data_path / uuid.json.
        tool_handlers : dict[str, ToolHandlerRegistration] or None, default None
            Mapping from tool names to handler registrations that include optional
            ToolSpec metadata to parse tool outputs by name. Each handler receives
            a ResponseFunctionToolCall and returns a string or any serializable
            result. Defaults to an empty dict when not provided.
        openai_settings : OpenAISettings or None, default None
            Fully configured OpenAI settings with API key and default model.
            Required for normal operation.

        Raises
        ------
        ValueError
            If api_key is missing from openai_settings.
            If default_model is missing from openai_settings.
        RuntimeError
            If the OpenAI client fails to initialize.

        Examples
        --------
        >>> from openai_sdk_helpers import ResponseBase, OpenAISettings
        >>> settings = OpenAISettings(api_key="sk-...", default_model="gpt-4")
        >>> response = ResponseBase(
        ...     name="my_session",
        ...     instructions="You are helpful",
        ...     tools=None,
        ...     output_structure=None,
        ...     tool_handlers={},
        ...     openai_settings=settings,
        ... )
        """
        if openai_settings is None:
            raise ValueError("openai_settings is required")

        self._tool_handlers = tool_handlers or {}
        self.uuid = uuid.uuid4()
        self._name = name

        # Resolve data_path with class name appended
        class_name = self.__class__.__name__
        if data_path is not None:
            data_path_obj = Path(data_path)
            if data_path_obj.name == class_name:
                self._data_path = data_path_obj
            else:
                self._data_path = data_path_obj / class_name
        else:
            from ..environment import get_data_path

            self._data_path = get_data_path(self.__class__.__name__)

        self._instructions = instructions
        self._tools: list[dict[str, Any]] | None = None
        if tools is not None:
            self._tools = [
                tool_handler.tool_spec.as_tool_definition() for tool_handler in tools
            ]

        self._output_structure = output_structure
        self._system_vector_store = system_vector_store
        self._openai_settings = openai_settings

        if not self._openai_settings.api_key:
            raise ValueError("OpenAI API key is required")

        self._client: OpenAIClient
        try:
            self._client = self._openai_settings.create_client()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Failed to initialize OpenAI client") from exc

        self._model = self._openai_settings.default_model
        if not self._model:
            raise ValueError(
                "OpenAI model is required. Set 'default_model' on OpenAISettings."
            )

        system_content: ResponseInputMessageContentListParam = [
            ResponseInputTextParam(type="input_text", text=instructions)
        ]

        self._user_vector_storage: Any | None = None

        # Initialize Files API manager for tracking uploaded files
        from ..files_api import FilesAPIManager

        self._files_manager = FilesAPIManager(self._client, auto_track=True)

        # New logic: system_vector_store is a list of vector store names to attach
        if system_vector_store:
            from .vector_store import attach_vector_store

            attach_vector_store(
                self,
                system_vector_store,
                api_key=(
                    self._client.api_key
                    if hasattr(self._client, "api_key")
                    else self._openai_settings.api_key
                ),
            )

            # Add retrieval guidance to system instructions to encourage RAG usage
            try:
                store_names = ", ".join(system_vector_store)
            except Exception:
                store_names = "attached vector stores"
            guidance_text = (
                "Retrieval guidance: You have access to a file_search tool "
                f"connected to vector store(s) {store_names}. When relevant, "
                "use file_search to retrieve supporting passages before answering. "
                "Cite or reference retrieved content when helpful."
            )
            system_content.append(
                ResponseInputTextParam(type="input_text", text=guidance_text)
            )

        self.messages = ResponseMessages()
        self.messages.add_system_message(content=system_content)
        if self._data_path is not None:
            self.save()

    @property
    def name(self) -> str:
        """Return the name of this response session.

        Returns
        -------
        str
            Name used for organizing artifacts and naming vector stores.

        Examples
        --------
        >>> response.name
        'my_session'
        """
        return self._name

    @property
    def instructions_text(self) -> str:
        """Return the system instructions for this response.

        Returns
        -------
        str
            System instructions provided to the OpenAI API.

        Examples
        --------
        >>> response.instructions_text
        'You are a helpful assistant.'
        """
        return self._instructions

    @property
    def tools(self) -> list[dict[str, Any]] | None:
        """Return the tool definitions for this response.

        Returns
        -------
        list or None
            Tool definitions provided to the OpenAI API, or None.

        Examples
        --------
        >>> response.tools
        []
        """
        return self._tools

    @property
    def output_structure(self) -> type[T] | None:
        """Return the output structure class for this response.

        Returns
        -------
        type[StructureBase] or None
            Structure class used to parse tool call outputs, or None.

        Examples
        --------
        >>> response.output_structure
        None
        """
        return self._output_structure

    def register_tool(
        self,
        func: Callable[..., Any],
        *,
        tool_spec: ToolSpec,
    ) -> None:
        """Register a tool handler and definition using a ToolSpec.

        Parameters
        ----------
        func : Callable[..., Any]
            Tool implementation function to wrap for argument parsing and
            result serialization.
        tool_spec : ToolSpec
            Tool specification describing input/output structures and metadata.

        Returns
        -------
        None
            Register the tool handler and definition for this response session.

        Raises
        ------
        ValueError
            If tool_spec.tool_name is empty.

        Examples
        --------
        >>> response.register_tool(run_search, tool_spec=search_tool_spec)
        """
        if not tool_spec.tool_name:
            raise ValueError("tool_spec.tool_name must be a non-empty string")
        self._tool_handlers[tool_spec.tool_name] = ToolHandlerRegistration(
            handler=func,
            tool_spec=tool_spec,
        )

    def _build_input(
        self,
        content: str | list[str],
        files: list[str] | None = None,
        use_vector_store: bool = False,
    ) -> None:
        """Construct input messages for the OpenAI API request.

        Automatically detects file types and handles them appropriately:
        - Images (jpg, png, gif, etc.) are sent as base64-encoded images
        - Documents are sent as base64-encoded file data by default
        - Documents can optionally be uploaded to vector stores for RAG

        Parameters
        ----------
        content : str or list[str]
            String or list of strings to include as user messages.
        files : list[str] or None, default None
            Optional list of file paths. Each file is automatically processed
            based on its type:
            - Images are base64-encoded as input_image
            - Documents are base64-encoded as input_file (default)
            - Documents can be uploaded to vector stores if use_vector_store=True
        use_vector_store : bool, default False
            If True, non-image files are uploaded to a vector store for
            RAG-enabled file search instead of inline base64 encoding.

        Notes
        -----
        When use_vector_store is True, this method automatically creates
        a vector store and adds a file_search tool for document retrieval.
        Images are always base64-encoded regardless of this setting.
        When multiple content strings are provided, file attachments are
        included only with the first message to avoid duplicating input
        files across messages.

        Examples
        --------
        >>> # Automatic handling - images as base64, docs inline
        >>> response._build_input("Analyze these", files=["photo.jpg", "doc.pdf"])

        >>> # Use vector store for documents (RAG)
        >>> response._build_input(
        ...     "Search these documents",
        ...     files=["doc1.pdf", "doc2.pdf"],
        ...     use_vector_store=True
        ... )
        """
        from .files import process_files

        contents = ensure_list(content)
        all_files = files or []

        # Process files using the dedicated files module
        vector_file_refs, base64_files, image_contents = process_files(
            self, all_files, use_vector_store
        )

        attachments: list[
            ResponseInputFileParam
            | ResponseInputFileContentParam
            | ResponseInputImageContentParam
        ] = []
        attachments.extend(vector_file_refs)
        attachments.extend(base64_files)
        attachments.extend(image_contents)

        # Add each content as a separate message.
        for index, raw_content in enumerate(contents):
            processed_text = raw_content.strip()
            input_content: list[
                ResponseInputTextParam
                | ResponseInputFileParam
                | ResponseInputFileContentParam
                | ResponseInputImageContentParam
            ] = [ResponseInputTextParam(type="input_text", text=processed_text)]

            if index == 0:
                input_content.extend(attachments)

            message = cast(
                ResponseInputItemParam,
                {"role": "user", "content": input_content},
            )
            self.messages.add_user_message(message)

    async def run_async(
        self,
        content: str | list[str],
        files: str | list[str] | None = None,
        use_vector_store: bool = False,
        save_messages: bool = True,
    ) -> T | str:
        """Generate a response asynchronously from the OpenAI API.

        Builds input messages, sends the request to OpenAI, processes any
        tool calls with registered handlers, and optionally parses the
        result into the configured output_structure.

        Automatically detects file types:
        - Images are sent as base64-encoded images
        - Documents are sent as base64-encoded files (default)
        - Documents can optionally use vector stores for RAG

        Parameters
        ----------
        content : str or list[str]
            Prompt text or list of prompt texts to send.
        files : str, list[str], or None, default None
            Optional file path or list of file paths. Each file is
            automatically processed based on its type.
        use_vector_store : bool, default False
            If True, non-image files are uploaded to a vector store
            for RAG-enabled search instead of inline base64 encoding.
        save_messages : bool, default True
            When True, persist the message history after each response or
            tool call.

        Returns
        -------
        T or str
            Parsed response object of type output_structure, or raw string if
            no structured output was produced.

        Raises
        ------
        RuntimeError
            If the API returns no output.
            If a tool handler raises an exception.
        ValueError
            If the API invokes a tool with no registered handler.

        Examples
        --------
        >>> # Automatic type detection
        >>> result = await response.run_async(
        ...     "Analyze these files",
        ...     files=["photo.jpg", "document.pdf"]
        ... )

        >>> # Use vector store for documents
        >>> result = await response.run_async(
        ...     "Search these documents",
        ...     files=["doc1.pdf", "doc2.pdf"],
        ...     use_vector_store=True
        ... )
        """
        log(f"{self.__class__.__name__}::run_response")
        parsed_result: T | None = None

        try:
            self._build_input(
                content=content,
                files=(ensure_list(files) if files else None),
                use_vector_store=use_vector_store,
            )

            kwargs = {
                "input": self.messages.to_openai_payload(),
                "model": self._model,
            }
            if not self._tools and self._output_structure is not None:
                kwargs["text"] = self._output_structure.response_format()

            if self._tools:
                kwargs["tools"] = self._tools
                kwargs["tool_choice"] = "auto"
            response = self._client.responses.create(**kwargs)

            if not response.output:
                log("No output returned from OpenAI.", level=logging.ERROR)
                raise RuntimeError("No output returned from OpenAI.")

            for response_output in response.output:
                if isinstance(response_output, ResponseFunctionToolCall):
                    log(
                        f"Tool call detected. Executing {response_output.name}.",
                        level=logging.INFO,
                    )

                    tool_name = response_output.name
                    registration = self._tool_handlers.get(tool_name)

                    if registration is None:
                        log(
                            f"No handler found for tool '{tool_name}'",
                            level=logging.ERROR,
                        )
                        raise ValueError(f"No handler for tool: {tool_name}")

                    handler = registration.handler
                    tool_spec = registration.tool_spec

                    try:
                        if inspect.iscoroutinefunction(handler):
                            tool_result_json = await handler(response_output)
                        else:
                            tool_result_json = handler(response_output)
                        if isinstance(tool_result_json, str):
                            tool_result = json.loads(tool_result_json)
                            tool_output = tool_result_json
                        else:
                            tool_result = coerce_jsonable(tool_result_json)
                            tool_output = json.dumps(tool_result, cls=customJSONEncoder)
                        self.messages.add_tool_message(
                            content=response_output, output=tool_output
                        )
                        if save_messages:
                            self.save()
                    except Exception as exc:
                        log(
                            f"Error executing tool handler '{tool_name}': {exc}",
                            level=logging.ERROR,
                            exc=exc,
                        )
                        raise RuntimeError(
                            f"Error in tool handler '{tool_name}': {exc}"
                        )

                    if tool_spec is not None:
                        output_dict = tool_spec.output_structure.from_json(tool_result)
                        parsed_result = cast(T, output_dict)
                    elif self._output_structure:
                        output_dict = self._output_structure.from_json(tool_result)
                        parsed_result = output_dict
                    else:
                        print(tool_result)
                        parsed_result = cast(T, tool_result)

                if isinstance(response_output, ResponseOutputMessage):
                    self.messages.add_assistant_message(
                        response_output, metadata=kwargs
                    )
                    if save_messages:
                        self.save()
                    if hasattr(response, "output_text") and response.output_text:
                        raw_text = response.output_text
                        log("No tool call. Parsing output_text.")
                        try:
                            output_dict = json.loads(raw_text)
                            if self._output_structure:
                                return self._output_structure.from_json(output_dict)
                            return output_dict
                        except Exception:
                            print(raw_text)
            if parsed_result is not None:
                return parsed_result
            return response.output_text
        except Exception as exc:
            try:
                self.save_error(exc)
            except Exception as save_exc:
                log(
                    f"Failed to save error details for response {self.uuid}: {save_exc}",
                    level=logging.ERROR,
                    exc=save_exc,
                )
            log(
                f"Error running response '{self._name}': {exc}",
                level=logging.ERROR,
                exc=exc,
            )
            raise

    def run_sync(
        self,
        content: str | list[str],
        *,
        files: str | list[str] | None = None,
        use_vector_store: bool = False,
        save_messages: bool = True,
    ) -> T | str:
        """Execute run_async synchronously with proper event loop handling.

        Automatically detects if an event loop is already running and uses
        a separate thread if necessary. This enables safe usage in both
        synchronous and asynchronous contexts.

        Automatically detects file types:
        - Images are sent as base64-encoded images
        - Documents are sent as base64-encoded files (default)
        - Documents can optionally use vector stores for RAG

        Parameters
        ----------
        content : str or list[str]
            Prompt text or list of prompt texts to send.
        files : str, list[str], or None, default None
            Optional file path or list of file paths. Each file is
            automatically processed based on its type.
        use_vector_store : bool, default False
            If True, non-image files are uploaded to a vector store
            for RAG-enabled search instead of inline base64 encoding.
        save_messages : bool, default True
            When True, persist the message history after each response or
            tool call.

        Returns
        -------
        T or str
            Parsed response object of type output_structure, or raw string.

        Raises
        ------
        RuntimeError
            If the API returns no output.
            If a tool handler raises an exception.
        ValueError
            If the API invokes a tool with no registered handler.

        Examples
        --------
        >>> # Automatic type detection
        >>> result = response.run_sync(
        ...     "Analyze these files",
        ...     files=["photo.jpg", "document.pdf"]
        ... )

        >>> # Use vector store for documents
        >>> result = response.run_sync(
        ...     "Search these documents",
        ...     files=["doc1.pdf", "doc2.pdf"],
        ...     use_vector_store=True
        ... )
        """

        async def runner() -> T | str:
            return await self.run_async(
                content=content,
                files=files,
                use_vector_store=use_vector_store,
                save_messages=save_messages,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())
        result: T | str = ""

        def _thread_func() -> None:
            nonlocal result
            result = asyncio.run(runner())

        thread = threading.Thread(target=_thread_func)
        thread.start()
        thread.join()
        return result

    def get_last_tool_message(self) -> ResponseMessage | None:
        """Return the most recent tool message from conversation history.

        Returns
        -------
        ResponseMessage or None
            Latest tool message, or None if no tool messages exist.

        Examples
        --------
        >>> message = response.get_last_tool_message()
        """
        return self.messages.get_last_tool_message()

    def get_last_user_message(self) -> ResponseMessage | None:
        """Return the most recent user message from conversation history.

        Returns
        -------
        ResponseMessage or None
            Latest user message, or None if no user messages exist.

        Examples
        --------
        >>> message = response.get_last_user_message()
        """
        return self.messages.get_last_user_message()

    def get_last_assistant_message(self) -> ResponseMessage | None:
        """Return the most recent assistant message from conversation history.

        Returns
        -------
        ResponseMessage or None
            Latest assistant message, or None if no assistant messages exist.

        Examples
        --------
        >>> message = response.get_last_assistant_message()
        """
        return self.messages.get_last_assistant_message()

    @classmethod
    def build_streamlit_config(
        cls: type[RB],
        *,
        display_title: str = "Example copilot",
        description: str | None = None,
        system_vector_store: Sequence[str] | str | None = None,
        preserve_vector_stores: bool = False,
        model: str | None = None,
    ) -> StreamlitAppConfig:
        """Construct a StreamlitAppConfig bound to this response class.

        Creates a complete Streamlit application configuration using the
        calling class as the response builder. This enables rapid deployment
        of chat interfaces for custom response classes.

        Parameters
        ----------
        display_title : str, default "Example copilot"
            Title displayed at the top of the Streamlit page.
        description : str or None, default None
            Optional description shown beneath the title.
        system_vector_store : Sequence[str], str, or None, default None
            Optional vector store name(s) to attach as system context.
            Single string or sequence of strings.
        preserve_vector_stores : bool, default False
            When True, skip automatic cleanup of vector stores on session close.
        model : str or None, default None
            Optional model identifier displayed in the chat interface.

        Returns
        -------
        StreamlitAppConfig
            Fully configured Streamlit application bound to this response class.

        Examples
        --------
        >>> configuration = MyResponse.build_streamlit_config(
        ...     display_title="My Assistant",
        ...     description="A helpful AI assistant",
        ...     system_vector_store=["docs", "kb"],
        ...     model="gpt-4"
        ... )
        """
        from openai_sdk_helpers.streamlit_app.configuration import StreamlitAppConfig

        normalized_stores = None
        if system_vector_store is not None:
            normalized_stores = ensure_list(system_vector_store)

        return StreamlitAppConfig(
            response=cls,
            display_title=display_title,
            description=description,
            system_vector_store=normalized_stores,
            preserve_vector_stores=preserve_vector_stores,
            model=model,
        )

    def save(self, filepath: str | Path | None = None) -> None:
        """Serialize the message history to a JSON file.

        Saves the complete conversation history to disk. The target path
        is determined by filepath parameter, or the configured data_path.

        Parameters
        ----------
        filepath : str, Path, or None, default None
            Optional explicit path for the JSON file. If None, constructs
            path from data_path and session UUID.

        Notes
        -----
        If no filepath is provided, the save operation writes to the
        session data path. If the configured data path already ends with
        the response name, it writes to data_path / uuid.json. Otherwise,
        it writes to data_path / name / uuid.json. The data path is
        configured during initialization and defaults to get_data_path().

        Raises
        ------
        IOError
            If the file cannot be written to disk.

        Examples
        --------
        >>> response.save("/path/to/session.json")
        >>> response.save()  # Uses data_path / uuid.json
        """
        if filepath is not None:
            target = Path(filepath)
        else:
            filename = f"{str(self.uuid).lower()}.json"
            target = self._session_path(filename)

        checked = check_filepath(filepath=target)
        self.messages.to_json_file(str(checked))
        log(f"Saved messages to {target}")

    def save_error(self, exc: BaseException) -> Path:
        """Persist error details to a file named with the response UUID.

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
        ...     response.run_sync("trigger error")
        ... except Exception as exc:
        ...     response.save_error(exc)
        """
        error_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        filename = f"{str(self.uuid).lower()}_error.txt"
        target = self._session_path(filename)
        checked = check_filepath(filepath=target)
        checked.write_text(error_text, encoding="utf-8")
        log(f"Saved error details to {checked}")
        return checked

    def _session_path(self, filename: str) -> Path:
        """Return the resolved session filepath for a given filename."""
        if self._data_path.name == self._name:
            return self._data_path / filename
        return self._data_path / self._name / filename

    def __repr__(self) -> str:
        """Return a detailed string representation of the response session.

        Returns
        -------
        str
            String showing class name, model, UUID, message count, and data path.
        """
        return (
            f"<{self.__class__.__name__}(model={self._model}, uuid={self.uuid}, "
            f"messages={len(self.messages.messages)}, data_path={self._data_path}>"
        )

    def __enter__(self) -> ResponseBase[T]:
        """Enter the context manager for resource management.

        Returns
        -------
        ResponseBase[T]
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
        """Clean up session resources including vector stores and uploaded files.

        Saves the current message history, deletes managed vector stores, and
        cleans up all tracked Files API uploads. User vector stores are always
        cleaned up. System vector store cleanup is handled via tool configuration.

        Notes
        -----
        This method is automatically called when using the response as a
        context manager. Always call close() or use a with statement to
        ensure proper resource cleanup.

        Examples
        --------
        >>> response = ResponseBase(...)
        >>> try:
        ...     result = response.run_sync("query")
        ... finally:
        ...     response.close()
        """
        log(f"Closing session {self.uuid} for {self.__class__.__name__}")
        self.save()

        # Clean up tracked Files API uploads
        try:
            if hasattr(self, "_files_manager") and self._files_manager:
                cleanup_results = self._files_manager.cleanup()
                if cleanup_results:
                    successful = sum(cleanup_results.values())
                    log(
                        f"Files API cleanup: {successful}/{len(cleanup_results)} files deleted"
                    )
        except Exception as exc:
            try:
                self.save_error(exc)
            except Exception as save_exc:
                log(
                    f"Failed to save error details for response {self.uuid}: {save_exc}",
                    level=logging.ERROR,
                    exc=save_exc,
                )
            log(
                f"Error cleaning up Files API uploads: {exc}",
                level=logging.WARNING,
                exc=exc,
            )

        # Always clean user vector storage if it exists
        try:
            if self._user_vector_storage:
                self._user_vector_storage.delete()
                log("User vector store deleted.")
        except Exception as exc:
            try:
                self.save_error(exc)
            except Exception as save_exc:
                log(
                    f"Failed to save error details for response {self.uuid}: {save_exc}",
                    level=logging.ERROR,
                    exc=save_exc,
                )
            log(
                f"Error deleting user vector store: {exc}",
                level=logging.WARNING,
                exc=exc,
            )
        # System vector store cleanup is now handled via tool configuration
        log(f"Session {self.uuid} closed.")

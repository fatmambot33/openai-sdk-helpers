"""Core response management for OpenAI API interactions.

This module implements the BaseResponse class, which manages the complete
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

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_param import ResponseInputItemParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_message import ResponseOutputMessage

from .messages import ResponseMessage, ResponseMessages
from ..config import OpenAISettings
from ..structure import BaseStructure
from ..types import OpenAIClient
from ..utils import ensure_list, log

if TYPE_CHECKING:  # pragma: no cover - only for typing hints
    from openai_sdk_helpers.streamlit_app.config import StreamlitAppConfig

T = TypeVar("T", bound=BaseStructure)
ToolHandler = Callable[[ResponseFunctionToolCall], str | Any]
ProcessContent = Callable[[str], tuple[str, list[str]]]


RB = TypeVar("RB", bound="BaseResponse[BaseStructure]")


class BaseResponse(Generic[T]):
    """Manage OpenAI API interactions for structured responses.

    Orchestrates the complete lifecycle of OpenAI API requests including
    input construction, tool execution, message history management, vector
    store attachments, and structured output parsing. Supports both
    synchronous and asynchronous execution with automatic resource cleanup.

    The class handles conversation state, tool calls with custom handlers,
    file attachments via vector stores, and optional parsing into typed
    structured output models. Sessions can be persisted to disk and restored.

    Attributes
    ----------
    uuid : UUID
        Unique identifier for this response session.
    name : str
        Lowercase class name used for path construction.
    messages : ResponseMessages
        Complete message history for this session.

    Methods
    -------
    run_async(content, attachments=None)
        Generate a response asynchronously and return parsed output.
    run_sync(content, attachments=None)
        Execute run_async synchronously with thread management.
    run_streamed(content, attachments=None)
        Execute run_async and await the result (streaming not yet supported).
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
    close()
        Clean up remote resources including vector stores.

    Examples
    --------
    >>> from openai_sdk_helpers import BaseResponse, OpenAISettings
    >>> settings = OpenAISettings(api_key="...", default_model="gpt-4")
    >>> response = BaseResponse(
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
        instructions: str,
        tools: list | None,
        output_structure: type[T] | None,
        tool_handlers: dict[str, ToolHandler],
        openai_settings: OpenAISettings,
        process_content: ProcessContent | None = None,
        name: str | None = None,
        system_vector_store: list[str] | None = None,
        data_path_fn: Callable[[str], Path] | None = None,
        save_path: Path | str | None = None,
    ) -> None:
        """Initialize a response session with OpenAI configuration.

        Sets up the OpenAI client, message history, vector stores, and tool
        handlers for a complete response workflow. The session can optionally
        be persisted to disk for later restoration.

        Parameters
        ----------
        instructions : str
            System instructions provided to the OpenAI API for context.
        tools : list or None
            Tool definitions for the OpenAI API request. Pass None for no tools.
        output_structure : type[BaseStructure] or None
            Structure class used to parse tool call outputs. When provided,
            the schema is automatically generated using the structure's
            response_format() method. Pass None for unstructured responses.
        tool_handlers : dict[str, ToolHandler]
            Mapping from tool names to callable handlers. Each handler receives
            a ResponseFunctionToolCall and returns a string or any serializable
            result.
        openai_settings : OpenAISettings
            Fully configured OpenAI settings with API key and default model.
        process_content : callable or None, default None
            Optional callback that processes input text and extracts file
            attachments. Must return a tuple of (processed_text, attachment_list).
        name : str or None, default None
            Module name used for data path construction when data_path_fn is set.
        system_vector_store : list[str] or None, default None
            Optional list of vector store names to attach as system context.
        data_path_fn : callable or None, default None
            Function mapping name to a base directory path for artifact storage.
        save_path : Path, str, or None, default None
            Optional path to a directory or file where message history is saved.
            If a directory, files are named using the session UUID.

        Raises
        ------
        ValueError
            If api_key is missing from openai_settings.
            If default_model is missing from openai_settings.
        RuntimeError
            If the OpenAI client fails to initialize.

        Examples
        --------
        >>> from openai_sdk_helpers import BaseResponse, OpenAISettings
        >>> settings = OpenAISettings(api_key="sk-...", default_model="gpt-4")
        >>> response = BaseResponse(
        ...     instructions="You are helpful",
        ...     tools=None,
        ...     output_structure=None,
        ...     tool_handlers={},
        ...     openai_settings=settings
        ... )
        """
        self._tool_handlers = tool_handlers
        self._process_content = process_content
        self._name = name
        self._data_path_fn = data_path_fn
        self._save_path = Path(save_path) if save_path is not None else None
        self._instructions = instructions
        self._tools = tools if tools is not None else []
        self._output_structure = output_structure
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

        self.uuid = uuid.uuid4()
        self.name = self.__class__.__name__.lower()

        system_content: ResponseInputMessageContentListParam = [
            ResponseInputTextParam(type="input_text", text=instructions)
        ]

        self._user_vector_storage: Any | None = None

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

        self.messages = ResponseMessages()
        self.messages.add_system_message(content=system_content)
        if self._save_path is not None or (
            self._data_path_fn is not None and self._name is not None
        ):
            self.save()

    @property
    def data_path(self) -> Path:
        """Return the directory for persisting session artifacts.

        Constructs a path using data_path_fn, name, class name, and the
        session name. Both data_path_fn and name must be set during
        initialization for this property to work.

        Returns
        -------
        Path
            Absolute path for persisting response artifacts and message history.

        Raises
        ------
        RuntimeError
            If data_path_fn or name were not provided during initialization.

        Examples
        --------
        >>> response.data_path
        PosixPath('/data/myapp/baseresponse/session_123')
        """
        if self._data_path_fn is None or self._name is None:
            raise RuntimeError(
                "data_path_fn and name are required to build data paths."
            )
        base_path = self._data_path_fn(self._name)
        return base_path / self.__class__.__name__.lower() / self.name

    def _build_input(
        self,
        content: str | list[str],
        attachments: list[str] | None = None,
    ) -> None:
        """Construct input messages for the OpenAI API request.

        Processes content through the optional process_content callback,
        uploads any file attachments to vector stores, and adds all
        messages to the conversation history.

        Parameters
        ----------
        content : str or list[str]
            String or list of strings to include as user messages.
        attachments : list[str] or None, default None
            Optional list of file paths to upload and attach to the message.

        Notes
        -----
        If attachments are provided and no user vector storage exists, this
        method automatically creates one and adds a file_search tool to
        the tools list.
        """
        contents = ensure_list(content)

        for raw_content in contents:
            if self._process_content is None:
                processed_text, content_attachments = raw_content, []
            else:
                processed_text, content_attachments = self._process_content(raw_content)
            input_content: list[ResponseInputTextParam | ResponseInputFileParam] = [
                ResponseInputTextParam(type="input_text", text=processed_text)
            ]

            all_attachments = (attachments or []) + content_attachments

            for file_path in all_attachments:
                if self._user_vector_storage is None:
                    from openai_sdk_helpers.vector_storage import VectorStorage

                    store_name = f"{self.__class__.__name__.lower()}_{self.name}_{self.uuid}_user"
                    self._user_vector_storage = VectorStorage(
                        store_name=store_name,
                        client=self._client,
                        model=self._model,
                    )
                    user_vector_storage = cast(Any, self._user_vector_storage)
                    if not any(
                        tool.get("type") == "file_search" for tool in self._tools
                    ):
                        self._tools.append(
                            {
                                "type": "file_search",
                                "vector_store_ids": [user_vector_storage.id],
                            }
                        )
                    else:
                        # If system vector store is attached, its ID will be in tool config
                        pass
                user_vector_storage = cast(Any, self._user_vector_storage)
                uploaded_file = user_vector_storage.upload_file(file_path)
                input_content.append(
                    ResponseInputFileParam(type="input_file", file_id=uploaded_file.id)
                )

            message = cast(
                ResponseInputItemParam,
                {"role": "user", "content": input_content},
            )
            self.messages.add_user_message(message)

    async def run_async(
        self,
        content: str | list[str],
        attachments: str | list[str] | None = None,
    ) -> T | None:
        """Generate a response asynchronously from the OpenAI API.

        Builds input messages, sends the request to OpenAI, processes any
        tool calls with registered handlers, and optionally parses the
        result into the configured output_structure.

        Parameters
        ----------
        content : str or list[str]
            Prompt text or list of prompt texts to send.
        attachments : str, list[str], or None, default None
            Optional file path or list of file paths to upload and attach.

        Returns
        -------
        T or None
            Parsed response object of type output_structure, or None if
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
        >>> result = await response.run_async("Analyze this text")
        >>> print(result)
        """
        log(f"{self.__class__.__name__}::run_response")
        parsed_result: T | None = None

        self._build_input(
            content=content,
            attachments=(ensure_list(attachments) if attachments else None),
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
                handler = self._tool_handlers.get(tool_name)

                if handler is None:
                    log(
                        f"No handler found for tool '{tool_name}'",
                        level=logging.ERROR,
                    )
                    raise ValueError(f"No handler for tool: {tool_name}")

                try:
                    if inspect.iscoroutinefunction(handler):
                        tool_result_json = await handler(response_output)
                    else:
                        tool_result_json = handler(response_output)
                    if isinstance(tool_result_json, str):
                        tool_result = json.loads(tool_result_json)
                        tool_output = tool_result_json
                    else:
                        tool_result = tool_result_json
                        tool_output = json.dumps(tool_result)
                    self.messages.add_tool_message(
                        content=response_output, output=tool_output
                    )
                    self.save()
                except Exception as exc:
                    log(
                        f"Error executing tool handler '{tool_name}': {exc}",
                        level=logging.ERROR,
                    )
                    raise RuntimeError(f"Error in tool handler '{tool_name}': {exc}")

                if self._output_structure:
                    output_dict = self._output_structure.from_raw_input(tool_result)
                    output_dict.console_print()
                    parsed_result = output_dict
                else:
                    print(tool_result)
                    parsed_result = cast(T, tool_result)

            if isinstance(response_output, ResponseOutputMessage):
                self.messages.add_assistant_message(response_output, kwargs)
                self.save()
                if hasattr(response, "output_text") and response.output_text:
                    raw_text = response.output_text
                    log("No tool call. Parsing output_text.")
                    try:
                        output_dict = json.loads(raw_text)
                        if self._output_structure:
                            return self._output_structure.from_raw_input(output_dict)
                        return output_dict
                    except Exception:
                        print(raw_text)
        if parsed_result is not None:
            return parsed_result
        return None

    def run_sync(
        self,
        content: str | list[str],
        attachments: str | list[str] | None = None,
    ) -> T | None:
        """Execute run_async synchronously with proper event loop handling.

        Automatically detects if an event loop is already running and uses
        a separate thread if necessary. This enables safe usage in both
        synchronous and asynchronous contexts.

        Parameters
        ----------
        content : str or list[str]
            Prompt text or list of prompt texts to send.
        attachments : str, list[str], or None, default None
            Optional file path or list of file paths to upload and attach.

        Returns
        -------
        T or None
            Parsed response object of type output_structure, or None.

        Examples
        --------
        >>> result = response.run_sync("Summarize this document")
        >>> print(result)
        """

        async def runner() -> T | None:
            return await self.run_async(content=content, attachments=attachments)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())
        result: T | None = None

        def _thread_func() -> None:
            nonlocal result
            result = asyncio.run(runner())

        thread = threading.Thread(target=_thread_func)
        thread.start()
        thread.join()
        return result

    def run_streamed(
        self,
        content: str | list[str],
        attachments: str | list[str] | None = None,
    ) -> T | None:
        """Execute run_async and await the result.

        Streaming responses are not yet fully supported, so this method
        simply awaits run_async to provide API compatibility with agent
        interfaces.

        Parameters
        ----------
        content : str or list[str]
            Prompt text or list of prompt texts to send.
        attachments : str, list[str], or None, default None
            Optional file path or list of file paths to upload and attach.

        Returns
        -------
        T or None
            Parsed response object of type output_structure, or None.

        Notes
        -----
        This method exists for API consistency but does not currently
        provide true streaming functionality.
        """
        return asyncio.run(self.run_async(content=content, attachments=attachments))

    def get_last_tool_message(self) -> ResponseMessage | None:
        """Return the most recent tool message from conversation history.

        Returns
        -------
        ResponseMessage or None
            Latest tool message, or None if no tool messages exist.
        """
        return self.messages.get_last_tool_message()

    def get_last_user_message(self) -> ResponseMessage | None:
        """Return the most recent user message from conversation history.

        Returns
        -------
        ResponseMessage or None
            Latest user message, or None if no user messages exist.
        """
        return self.messages.get_last_user_message()

    def get_last_assistant_message(self) -> ResponseMessage | None:
        """Return the most recent assistant message from conversation history.

        Returns
        -------
        ResponseMessage or None
            Latest assistant message, or None if no assistant messages exist.
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
        >>> config = MyResponse.build_streamlit_config(
        ...     display_title="My Assistant",
        ...     description="A helpful AI assistant",
        ...     system_vector_store=["docs", "kb"],
        ...     model="gpt-4"
        ... )
        """
        from openai_sdk_helpers.streamlit_app.config import StreamlitAppConfig

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
        is determined by filepath parameter, save_path from initialization,
        or data_path_fn if configured.

        Parameters
        ----------
        filepath : str, Path, or None, default None
            Optional explicit path for the JSON file. If None, uses save_path
            or constructs path from data_path_fn and session UUID.

        Notes
        -----
        If no save location is configured (no filepath, save_path, or
        data_path_fn), the save operation is silently skipped.

        Examples
        --------
        >>> response.save("/path/to/session.json")
        >>> response.save()  # Uses configured save_path or data_path
        """
        if filepath is not None:
            target = Path(filepath)
        elif self._save_path is not None:
            if self._save_path.suffix == ".json":
                target = self._save_path
            else:
                filename = f"{str(self.uuid).lower()}.json"
                target = self._save_path / filename
        elif self._data_path_fn is not None and self._name is not None:
            filename = f"{str(self.uuid).lower()}.json"
            target = self.data_path / filename
        else:
            log(
                "Skipping save: no filepath, save_path, or data_path_fn configured.",
                level=logging.DEBUG,
            )
            return

        self.messages.to_json_file(str(target))
        log(f"Saved messages to {target}")

    def __repr__(self) -> str:
        """Return a detailed string representation of the response session.

        Returns
        -------
        str
            String showing class name, model, UUID, message count, and data path.
        """
        data_path = None
        if self._data_path_fn is not None and self._name is not None:
            data_path = self.data_path
        return (
            f"<{self.__class__.__name__}(model={self._model}, uuid={self.uuid}, "
            f"messages={len(self.messages.messages)}, data_path={data_path}>"
        )

    def __enter__(self) -> BaseResponse[T]:
        """Enter the context manager for resource management.

        Returns
        -------
        BaseResponse[T]
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
        """Clean up session resources including vector stores.

        Saves the current message history and deletes managed vector stores.
        User vector stores are always cleaned up. System vector store cleanup
        is handled via tool configuration.

        Notes
        -----
        This method is automatically called when using the response as a
        context manager. Always call close() or use a with statement to
        ensure proper resource cleanup.

        Examples
        --------
        >>> response = BaseResponse(...)
        >>> try:
        ...     result = response.run_sync("query")
        ... finally:
        ...     response.close()
        """
        log(f"Closing session {self.uuid} for {self.__class__.__name__}")
        self.save()
        # Always clean user vector storage if it exists
        try:
            if self._user_vector_storage:
                self._user_vector_storage.delete()
                log("User vector store deleted.")
        except Exception as exc:
            log(f"Error deleting user vector store: {exc}", level=logging.WARNING)
        # System vector store cleanup is now handled via tool configuration
        log(f"Session {self.uuid} closed.")

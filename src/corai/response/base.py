"""Base response handling for OpenAI interactions."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from openai import OpenAI
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_param import ResponseInputItemParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_message import ResponseOutputMessage

from .messages import ResponseMessages
from ..structure import BaseStructure
from ..utils import ensure_list, log

T = TypeVar("T", bound=BaseStructure)
ToolHandler = Callable[[ResponseFunctionToolCall], Union[str, Any]]
ProcessContent = Callable[[str], Tuple[str, List[str]]]


class ResponseBase(Generic[T]):
    """Manage OpenAI interactions for structured responses.

    This base class handles input construction, OpenAI requests, tool calls,
    and optional parsing into structured output models.

    Methods
    -------
    generate_response_async(content, attachments)
        Generate a response asynchronously and return parsed output.
    generate_response(content, attachments)
        Synchronous wrapper around ``generate_response_async``.
    save(filepath)
        Serialize the message history to disk.
    close()
        Clean up remote resources (vector stores).
    """

    def __init__(
        self,
        *,
        instructions: str,
        tools: Optional[list],
        schema: Optional[Any],
        output_structure: Optional[Type[T]],
        tool_handlers: dict[str, ToolHandler],
        process_content: Optional[ProcessContent] = None,
        module_name: Optional[str] = None,
        vector_storage_cls: Optional[type] = None,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        attachments: Optional[Union[Tuple[str, str], list[Tuple[str, str]]]] = None,
        data_path_fn: Optional[Callable[[str], Path]] = None,
        save_path: Optional[Path | str] = None,
    ) -> None:
        """Initialize a response session.

        Parameters
        ----------
        instructions : str
            System instructions for the OpenAI response.
        tools : list or None
            Tool definitions for the OpenAI request.
        schema : object or None
            Optional response schema configuration.
        output_structure : type[BaseStructure] or None
            Structure type used to parse tool call outputs.
        tool_handlers : dict[str, ToolHandler]
            Mapping of tool names to handler callables.
        process_content : callable, optional
            Callback that cleans input text and extracts attachments.
        module_name : str, optional
            Module name used to build the data path.
        vector_storage_cls : type, optional
            Vector storage class used for file uploads.
        client : OpenAI or None, default=None
            Optional pre-initialized OpenAI client.
        model : str or None, default=None
            Optional OpenAI model name override.
        api_key : str or None, default=None
            Optional OpenAI API key override.
        attachments : tuple or list of tuples, optional
            File attachments in the form ``(file_path, tool_type)``.
        data_path_fn : callable or None, default=None
            Function that maps ``module_name`` to a base data path.
        save_path : Path | str or None, default=None
            Optional path to a directory or file for persisted messages.

        Raises
        ------
        ValueError
            If API key or model is missing.
        RuntimeError
            If the OpenAI client fails to initialize.
        """
        self._tool_handlers = tool_handlers
        self._process_content = process_content
        self._module_name = module_name
        self._vector_storage_cls = vector_storage_cls
        self._data_path_fn = data_path_fn
        self._save_path = Path(save_path) if save_path is not None else None
        self._instructions = instructions
        self._tools = tools if tools is not None else []
        self._schema = schema
        self._output_structure = output_structure

        if client is None:
            if api_key is None:
                raise ValueError("OpenAI API key is required")
            try:
                self._client = OpenAI(api_key=api_key)
            except Exception as exc:
                raise RuntimeError("Failed to initialize OpenAI client") from exc
        else:
            self._client = client

        self._model = model
        if not self._model:
            raise ValueError("OpenAI model is required")

        self.uuid = uuid.uuid4()
        self.name = self.__class__.__name__.lower()

        system_content: ResponseInputMessageContentListParam = [
            ResponseInputTextParam(type="input_text", text=instructions)
        ]

        self._system_vector_storage: Optional[Any] = None
        self._user_vector_storage: Optional[Any] = None

        if attachments:
            if self._vector_storage_cls is None:
                raise RuntimeError("vector_storage_cls is required for attachments.")
            self.file_objects: dict[str, List[str]] = {}
            storage_name = f"{self.__class__.__name__.lower()}_{self.name}_system"
            self._system_vector_storage = self._vector_storage_cls(
                store_name=storage_name, client=self._client, model=self._model
            )
            system_vector_storage = cast(Any, self._system_vector_storage)
            for file_path, tool_type in attachments:
                uploaded_file = system_vector_storage.upload_file(file_path=file_path)
                self.file_objects.setdefault(tool_type, []).append(uploaded_file.id)

            self.tool_resources = {}
            required_tools = []

            for tool_type, file_ids in self.file_objects.items():
                required_tools.append({"type": tool_type})
                self.tool_resources[tool_type] = {"file_ids": file_ids}
                if tool_type == "file_search":
                    self.tool_resources[tool_type]["vector_store_ids"] = [
                        system_vector_storage.id
                    ]

            existing_tool_types = {tool["type"] for tool in self._tools}
            for tool in required_tools:
                tool_type = tool["type"]
                if tool_type == "file_search":
                    tool["vector_store_ids"] = [system_vector_storage.id]
                if tool_type not in existing_tool_types:
                    self._tools.append(tool)

        self.messages = ResponseMessages()
        self.messages.add_system_message(content=system_content)
        if self._save_path is not None or (
            self._data_path_fn is not None and self._module_name is not None
        ):
            self.save()

    @property
    def data_path(self) -> Path:
        """Return the directory used to persist artifacts for this session.

        Returns
        -------
        Path
            Absolute path for persisting response artifacts.
        """
        if self._data_path_fn is None or self._module_name is None:
            raise RuntimeError(
                "data_path_fn and module_name are required to build data paths."
            )
        base_path = self._data_path_fn(self._module_name)
        return base_path / self.__class__.__name__.lower() / self.name

    def _build_input(
        self,
        content: Union[str, List[str]],
        attachments: Optional[List[str]] = None,
    ) -> None:
        """Build the list of input messages for the OpenAI request.

        Parameters
        ----------
        content
            String or list of strings to include as user messages.
        attachments
            Optional list of file paths to upload and attach.
        """
        contents = ensure_list(content)

        for raw_content in contents:
            if self._process_content is None:
                processed_text, content_attachments = raw_content, []
            else:
                processed_text, content_attachments = self._process_content(raw_content)
            input_content: List[
                Union[ResponseInputTextParam, ResponseInputFileParam]
            ] = [ResponseInputTextParam(type="input_text", text=processed_text)]

            all_attachments = (attachments or []) + content_attachments

            for file_path in all_attachments:
                if self._user_vector_storage is None:
                    if self._vector_storage_cls is None:
                        raise RuntimeError(
                            "vector_storage_cls is required for attachments."
                        )
                    store_name = f"{self.__class__.__name__.lower()}_{self.name}_{self.uuid}_user"
                    self._user_vector_storage = self._vector_storage_cls(
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
                        for tool in self._tools:
                            if tool.get("type") == "file_search":
                                if self._system_vector_storage is not None:
                                    tool["vector_store_ids"] = [
                                        cast(Any, self._system_vector_storage).id,
                                        user_vector_storage.id,
                                    ]
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

    async def generate_response_async(
        self,
        content: Union[str, List[str]],
        attachments: Optional[Union[str, List[str]]] = None,
    ) -> Optional[T]:
        """Generate a response asynchronously.

        Parameters
        ----------
        content
            Prompt text or list of texts.
        attachments
            Optional file path or list of paths to upload and attach.

        Returns
        -------
        Optional[T]
            Parsed response object or ``None``.

        Raises
        ------
        RuntimeError
            If the API returns no output or a tool handler errors.
        ValueError
            If no handler is found for a tool invoked by the API.
        """
        log(f"{self.__class__.__name__}::generate_response")
        parsed_result: Optional[T] = None

        self._build_input(
            content=content,
            attachments=(ensure_list(attachments) if attachments else None),
        )

        kwargs = {
            "input": self.messages.to_openai_payload(),
            "model": self._model,
        }
        if self._schema is not None:
            kwargs["text"] = self._schema

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
                        if self._output_structure and self._schema:
                            return self._output_structure.from_raw_input(output_dict)
                        return output_dict
                    except Exception:
                        print(raw_text)
        if parsed_result is not None:
            return parsed_result
        return None

    def generate_response(
        self,
        content: Union[str, List[str]],
        attachments: Optional[Union[str, List[str]]] = None,
    ) -> Optional[T]:
        """Run :meth:`generate_response_async` synchronously."""

        async def runner() -> Optional[T]:
            return await self.generate_response_async(
                content=content, attachments=attachments
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())
        result: Optional[T] = None

        def _thread_func() -> None:
            nonlocal result
            result = asyncio.run(runner())

        thread = threading.Thread(target=_thread_func)
        thread.start()
        thread.join()
        return result

    def save(self, filepath: Optional[str | Path] = None) -> None:
        """Serialize the message history to a JSON file."""
        if filepath is not None:
            target = Path(filepath)
        elif self._save_path is not None:
            if self._save_path.suffix == ".json":
                target = self._save_path
            else:
                filename = f"{str(self.uuid).lower()}.json"
                target = self._save_path / filename
        elif self._data_path_fn is not None and self._module_name is not None:
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
        """Return an unambiguous representation including model and UUID."""
        data_path = None
        if self._data_path_fn is not None and self._module_name is not None:
            data_path = self.data_path
        return (
            f"<{self.__class__.__name__}(model={self._model}, uuid={self.uuid}, "
            f"messages={len(self.messages.messages)}, data_path={data_path}>"
        )

    def __enter__(self) -> "ResponseBase[T]":
        """Enter the context manager for this response session."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and close remote resources."""
        self.close()

    def close(self) -> None:
        """Delete remote vector stores and clean up the session."""
        log(f"Closing session {self.uuid} for {self.__class__.__name__}")

        try:
            if self._user_vector_storage:
                self._user_vector_storage.delete()
                log("User vector store deleted.")
        except Exception as exc:
            log(f"Error deleting user vector store: {exc}", level=logging.WARNING)
        try:
            if self._system_vector_storage:
                self._system_vector_storage.delete()
                log("System vector store deleted.")
        except Exception as exc:
            log(f"Error deleting system vector store: {exc}", level=logging.WARNING)
        log(f"Session {self.uuid} closed.")

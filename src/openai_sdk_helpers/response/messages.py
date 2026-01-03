"""Message containers for OpenAI response conversations.

This module provides dataclasses for managing conversation history including
user inputs, assistant outputs, system messages, and tool calls. Messages are
stored with timestamps and metadata, and can be serialized to JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import cast

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputItemParam,
)
from openai.types.responses.response_output_message import ResponseOutputMessage

from ..utils import JSONSerializable
from .tool_call import ResponseToolCall


@dataclass
class ResponseMessage(JSONSerializable):
    """Single message exchanged with the OpenAI API.

    Represents a complete message with role, content, timestamp, and
    optional metadata. Can be serialized to JSON for persistence.

    Attributes
    ----------
    role : str
        Message role: "user", "assistant", "tool", or "system".
    content : ResponseInputItemParam | ResponseOutputMessage | ResponseFunctionToolCallParam | FunctionCallOutput | ResponseInputMessageContentListParam
        Message content in OpenAI format.
    timestamp : datetime
        UTC timestamp when the message was created.
    metadata : dict[str, str | float | bool]
        Optional metadata for tracking or debugging.

    Methods
    -------
    to_openai_format()
        Return the message content in OpenAI API format.
    """

    role: str  # "user", "assistant", "tool", etc.
    content: (
        ResponseInputItemParam
        | ResponseOutputMessage
        | ResponseFunctionToolCallParam
        | FunctionCallOutput
        | ResponseInputMessageContentListParam
    )
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, str | float | bool] = field(default_factory=dict)

    def to_openai_format(
        self,
    ) -> (
        ResponseInputItemParam
        | ResponseOutputMessage
        | ResponseFunctionToolCallParam
        | FunctionCallOutput
        | ResponseInputMessageContentListParam
    ):
        """Return the message in the format expected by the OpenAI client.

        Returns
        -------
        ResponseInputItemParam | ResponseOutputMessage | ResponseFunctionToolCallParam | FunctionCallOutput | ResponseInputMessageContentListParam
            Stored message content in OpenAI format.
        """
        return self.content


@dataclass
class ResponseMessages(JSONSerializable):
    """Collection of messages in a conversation.

    Manages the complete history of messages exchanged during an OpenAI
    API interaction. Provides methods for adding different message types
    and converting to formats required by the OpenAI API.

    Attributes
    ----------
    messages : list[ResponseMessage]
        Ordered list of all messages in the conversation.

    Methods
    -------
    add_system_message(content, **metadata)
        Append a system message to the conversation.
    add_user_message(input_content, **metadata)
        Append a user message to the conversation.
    add_assistant_message(content, metadata)
        Append an assistant message to the conversation.
    add_tool_message(content, output, **metadata)
        Record a tool call and its output.
    to_openai_payload()
        Convert stored messages to OpenAI input payload format.
    get_last_assistant_message()
        Return the most recent assistant message or None.
    get_last_tool_message()
        Return the most recent tool message or None.
    get_last_user_message()
        Return the most recent user message or None.
    """

    messages: list[ResponseMessage] = field(default_factory=list)

    def add_system_message(
        self, content: ResponseInputMessageContentListParam, **metadata
    ) -> None:
        """Append a system message to the conversation.

        Parameters
        ----------
        content : ResponseInputMessageContentListParam
            System message content in OpenAI format.
        **metadata
            Optional metadata to store with the message.
        """
        response_input = cast(
            ResponseInputItemParam, {"role": "system", "content": content}
        )
        self.messages.append(
            ResponseMessage(role="system", content=response_input, metadata=metadata)
        )

    def add_user_message(
        self, input_content: ResponseInputItemParam, **metadata
    ) -> None:
        """Append a user message to the conversation.

        Parameters
        ----------
        input_content : ResponseInputItemParam
            Message payload supplied by the user.
        **metadata
            Optional metadata to store with the message.
        """
        self.messages.append(
            ResponseMessage(role="user", content=input_content, metadata=metadata)
        )

    def add_assistant_message(
        self,
        content: ResponseOutputMessage,
        metadata: dict[str, str | float | bool],
    ) -> None:
        """Append an assistant message to the conversation.

        Parameters
        ----------
        content : ResponseOutputMessage
            Assistant response message from the OpenAI API.
        metadata : dict[str, str | float | bool]
            Optional metadata to store with the message.
        """
        self.messages.append(
            ResponseMessage(role="assistant", content=content, metadata=metadata)
        )

    def add_tool_message(
        self, content: ResponseFunctionToolCall, output: str, **metadata
    ) -> None:
        """Record a tool call and its output in the conversation.

        Parameters
        ----------
        content : ResponseFunctionToolCall
            Tool call received from the OpenAI API.
        output : str
            JSON string returned by the executed tool handler.
        **metadata
            Optional metadata to store with the message.
        """
        tool_call = ResponseToolCall(
            call_id=content.call_id,
            name=content.name,
            arguments=content.arguments,
            output=output,
        )
        function_call, function_call_output = tool_call.to_response_input_item_param()
        self.messages.append(
            ResponseMessage(role="tool", content=function_call, metadata=metadata)
        )
        self.messages.append(
            ResponseMessage(
                role="tool", content=function_call_output, metadata=metadata
            )
        )

    def to_openai_payload(
        self,
    ) -> list[
        ResponseInputItemParam
        | ResponseOutputMessage
        | ResponseFunctionToolCallParam
        | FunctionCallOutput
        | ResponseInputMessageContentListParam
    ]:
        """Convert stored messages to OpenAI API input format.

        Returns
        -------
        list
            List of message payloads suitable for the OpenAI API.
            Assistant messages are excluded as they are outputs, not inputs.

        Notes
        -----
        Assistant messages are not included in the returned payload since
        they represent model outputs rather than inputs for the next request.
        """
        return [
            msg.to_openai_format() for msg in self.messages if msg.role != "assistant"
        ]

    def _get_last_message(self, role: str) -> ResponseMessage | None:
        """Return the most recent message for the given role.

        Parameters
        ----------
        role : str
            Role name to filter messages by.

        Returns
        -------
        ResponseMessage or None
            Latest message matching ``role`` or ``None`` when absent.
        """
        for message in reversed(self.messages):
            if message.role == role:
                return message
        return None

    def get_last_assistant_message(self) -> ResponseMessage | None:
        """Return the most recent assistant message.

        Returns
        -------
        ResponseMessage or None
            Latest assistant message or ``None`` when absent.
        """
        return self._get_last_message(role="assistant")

    def get_last_tool_message(self) -> ResponseMessage | None:
        """Return the most recent tool message.

        Returns
        -------
        ResponseMessage or None
            Latest tool message or ``None`` when absent.
        """
        return self._get_last_message(role="tool")

    def get_last_user_message(self) -> ResponseMessage | None:
        """Return the most recent user message.

        Returns
        -------
        ResponseMessage or None
            Latest user message or ``None`` when absent.
        """
        return self._get_last_message(role="user")

"""Message containers for shared OpenAI responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Union, cast

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
    """Single message exchanged with the OpenAI client.

    Methods
    -------
    to_openai_format()
        Return the payload in the format expected by the OpenAI client.
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
    metadata: Dict[str, Union[str, float, bool]] = field(default_factory=dict)

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
    """Represent a collection of messages in a response.

    This dataclass encapsulates user inputs and assistant outputs during an
    OpenAI API interaction.

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
        Convert stored messages to the OpenAI input payload.
    """

    messages: List[ResponseMessage] = field(default_factory=list)

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

        Returns
        -------
        None
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

        Returns
        -------
        None
        """
        self.messages.append(
            ResponseMessage(role="user", content=input_content, metadata=metadata)
        )

    def add_assistant_message(
        self,
        content: ResponseOutputMessage,
        metadata: Dict[str, Union[str, float, bool]],
    ) -> None:
        """Append an assistant message to the conversation.

        Parameters
        ----------
        content : ResponseOutputMessage
            Assistant response message.
        metadata : dict[str, Union[str, float, bool]]
            Optional metadata to store with the message.

        Returns
        -------
        None
        """
        self.messages.append(
            ResponseMessage(role="assistant", content=content, metadata=metadata)
        )

    def add_tool_message(
        self, content: ResponseFunctionToolCall, output: str, **metadata
    ) -> None:
        """Record a tool call and its output in the conversation history.

        Parameters
        ----------
        content : ResponseFunctionToolCall
            Tool call received from OpenAI.
        output : str
            JSON string returned by the executed tool.
        **metadata
            Optional metadata to store with the message.

        Returns
        -------
        None
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
    ) -> List[
        ResponseInputItemParam
        | ResponseOutputMessage
        | ResponseFunctionToolCallParam
        | FunctionCallOutput
        | ResponseInputMessageContentListParam
    ]:
        """Convert stored messages to the input payload expected by OpenAI.

        Notes
        -----
        Assistant messages are model outputs and are not included in the
        next request's input payload.

        Returns
        -------
        list
            List of message payloads excluding assistant outputs.
        """
        return [
            msg.to_openai_format() for msg in self.messages if msg.role != "assistant"
        ]

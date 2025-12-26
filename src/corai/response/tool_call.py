"""Tool call representation for shared responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput


@dataclass
class ResponseToolCall:
    """Container for tool call data used in a conversation.

    Attributes
    ----------
    call_id : str
        Identifier of the tool call.
    name : str
        Name of the tool invoked.
    arguments : str
        JSON string with the arguments passed to the tool.
    output : str
        JSON string representing the result produced by the tool.

    Methods
    -------
    to_response_input_item_param()
        Convert stored data into OpenAI tool call objects.
    """

    call_id: str
    name: str
    arguments: str
    output: str

    def to_response_input_item_param(
        self,
    ) -> Tuple[ResponseFunctionToolCallParam, FunctionCallOutput]:
        """Convert stored data into OpenAI tool call objects.

        Returns
        -------
        tuple[ResponseFunctionToolCallParam, FunctionCallOutput]
            The function call object and the corresponding output object
            suitable for inclusion in an OpenAI request.
        """
        from typing import cast

        function_call = cast(
            ResponseFunctionToolCallParam,
            {
                "arguments": self.arguments,
                "call_id": self.call_id,
                "name": self.name,
                "type": "function_call",
            },
        )
        function_call_output = cast(
            FunctionCallOutput,
            {
                "call_id": self.call_id,
                "output": self.output,
                "type": "function_call_output",
            },
        )
        return function_call, function_call_output

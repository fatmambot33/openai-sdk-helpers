"""Response handling for OpenAI API interactions.

This module provides comprehensive support for managing OpenAI API responses,
including message handling, tool execution, vector store attachments, and
structured output parsing. It serves as the foundation for building
sophisticated AI agents with persistent conversation state.

Classes
-------
BaseResponse
    Core response manager for OpenAI interactions with structured outputs.
ResponseConfiguration
    Immutable configuration for defining request/response structures.
ResponseMessage
    Single message exchanged with the OpenAI client.
ResponseMessages
    Collection of messages in a response conversation.
ResponseToolCall
    Container for tool call data and formatting.

Functions
---------
run_sync
    Execute a response workflow synchronously with resource cleanup.
run_async
    Execute a response workflow asynchronously with resource cleanup.
run_streamed
    Execute a response workflow and return the asynchronous result.
attach_vector_store
    Attach vector stores to a response's file_search tool.
"""

from __future__ import annotations

from .base import BaseResponse
from .config import ResponseConfiguration, ResponseRegistry, get_default_registry
from .messages import ResponseMessage, ResponseMessages
from .runner import run_async, run_streamed, run_sync
from .tool_call import ResponseToolCall, parse_tool_arguments
from .vector_store import attach_vector_store

__all__ = [
    "BaseResponse",
    "ResponseConfiguration",
    "ResponseRegistry",
    "get_default_registry",
    "ResponseMessage",
    "ResponseMessages",
    "run_sync",
    "run_async",
    "run_streamed",
    "ResponseToolCall",
    "parse_tool_arguments",
    "attach_vector_store",
]

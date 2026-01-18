"""Response handling for OpenAI API interactions.

This module provides comprehensive support for managing OpenAI API responses,
including message handling, tool execution, vector store attachments, file
processing, and structured output parsing. It serves as the foundation for
building sophisticated AI agents with persistent conversation state.

Classes
-------
ResponseBase
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
attach_vector_store
    Attach vector stores to a response's file_search tool.
process_files
    Process file attachments with automatic type detection.
"""

from __future__ import annotations

from .base import ResponseBase
from .configuration import ResponseConfiguration, ResponseRegistry, get_default_registry
from .files import process_files
from .messages import ResponseMessage, ResponseMessages
from .runner import run_async, run_sync
from .tool_call import ResponseToolCall
from .vector_store import attach_vector_store

__all__ = [
    "ResponseBase",
    "ResponseConfiguration",
    "ResponseRegistry",
    "get_default_registry",
    "ResponseMessage",
    "ResponseMessages",
    "run_sync",
    "run_async",
    "ResponseToolCall",
    "attach_vector_store",
    "process_files",
]

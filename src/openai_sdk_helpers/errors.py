"""Custom exception hierarchy for openai-sdk-helpers.

Provides specific exception types for different error scenarios,
improving error handling and debugging capabilities.
"""

from collections.abc import Mapping


class OpenAISDKError(Exception):
    """Base exception for openai-sdk-helpers library.

    All custom exceptions in this library inherit from this class,
    allowing callers to catch all SDK-specific errors.

    Parameters
    ----------
    message : str
        Human-readable error message
    context : Mapping[str, object] | None
        Additional context information for debugging. Default is None.

    Examples
    --------
    >>> try:
    ...     raise OpenAISDKError("Something went wrong", context={"step": "init"})
    ... except OpenAISDKError as exc:
    ...     print(f"SDK Error: {exc}")
    ...     print(f"Context: {exc.context}")
    """

    def __init__(
        self,
        message: str,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the exception with message and optional context."""
        super().__init__(message)
        self.context = dict(context) if context is not None else {}


class ConfigurationError(OpenAISDKError):
    """Configuration validation or initialization failed.

    Raised when configuration is missing, invalid, or inconsistent.
    """

    pass


class PromptNotFoundError(OpenAISDKError):
    """Prompt template file not found or cannot be read.

    Raised when a required prompt template file is missing or inaccessible.
    """

    pass


class AgentExecutionError(OpenAISDKError):
    """Agent execution failed.

    Raised when an agent encounters an error during execution.
    May wrap underlying exceptions with additional context.
    """

    pass


class VectorStorageError(OpenAISDKError):
    """Vector storage operation failed.

    Raised when vector store operations (upload, download, cleanup) fail.
    """

    pass


class ToolExecutionError(OpenAISDKError):
    """Tool execution failed.

    Raised when a tool handler encounters an error.
    """

    pass


class ResponseGenerationError(OpenAISDKError):
    """Response generation failed.

    Raised when generating responses from structured output fails.
    """

    pass


class InputValidationError(OpenAISDKError):
    """Input validation failed.

    Raised when provided input doesn't meet required constraints.
    """

    pass


class AsyncExecutionError(OpenAISDKError):
    """Asynchronous operation failed.

    Raised when async/await operations fail or timeout.
    """

    pass


class ResourceCleanupError(OpenAISDKError):
    """Resource cleanup failed.

    Raised when cleanup of resources fails, but may not be fatal.
    """

    pass


class ExtractionError(OpenAISDKError):
    """Extraction execution failed.

    Raised when LangExtract operations fail or output validation fails.
    """

    pass

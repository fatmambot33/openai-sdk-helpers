"""Tests for exception hierarchy module."""

import pytest

from openai_sdk_helpers.errors import (
    OpenAISDKError,
    ConfigurationError,
    PromptNotFoundError,
    AgentExecutionError,
    VectorStorageError,
    ToolExecutionError,
    ResponseGenerationError,
    InputValidationError,
    AsyncExecutionError,
    ResourceCleanupError,
)


class TestExceptionHierarchy:
    """Test custom exception classes."""

    def test_base_exception_inherits_from_exception(self) -> None:
        """OpenAISDKError should inherit from Exception."""
        assert issubclass(OpenAISDKError, Exception)

    def test_specific_exceptions_inherit_from_base(self) -> None:
        """All specific exceptions should inherit from OpenAISDKError."""
        exceptions = [
            ConfigurationError,
            PromptNotFoundError,
            AgentExecutionError,
            VectorStorageError,
            ToolExecutionError,
            ResponseGenerationError,
            InputValidationError,
            AsyncExecutionError,
            ResourceCleanupError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, OpenAISDKError)

    def test_exception_with_message(self) -> None:
        """Exception should store message."""
        msg = "Test error message"
        exc = OpenAISDKError(msg)
        assert str(exc) == msg

    def test_exception_with_context(self) -> None:
        """Exception should store context dict."""
        context = {"key": "value", "step": 1}
        exc = OpenAISDKError("Error", context=context)
        assert exc.context == context

    def test_configuration_error(self) -> None:
        """ConfigurationError should work correctly."""
        exc = ConfigurationError("Missing config")
        assert isinstance(exc, OpenAISDKError)
        assert isinstance(exc, ConfigurationError)

    def test_async_execution_error(self) -> None:
        """AsyncExecutionError should work correctly."""
        exc = AsyncExecutionError("Timeout", context={"timeout": 30})
        assert exc.context == {"timeout": 30}
        assert isinstance(exc, OpenAISDKError)

    def test_exception_catching_by_base(self) -> None:
        """Specific exceptions should be catchable by base exception."""
        with pytest.raises(OpenAISDKError):
            raise ConfigurationError("test")

    def test_exception_with_chain(self) -> None:
        """Exception should support chaining."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            with pytest.raises(AgentExecutionError) as exc_info:
                raise AgentExecutionError("Agent failed") from e
            assert exc_info.value.__cause__ is not None

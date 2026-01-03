"""Tests for logging configuration module."""

import logging

import pytest

from openai_sdk_helpers.logging_config import LoggerFactory


class TestLoggerFactory:
    """Test centralized logging configuration."""

    def test_configure_with_default_level(self) -> None:
        """Should configure logging with default INFO level."""
        LoggerFactory.configure()
        logger = LoggerFactory.get_logger("test")
        assert logger.level == logging.INFO or logger.level == 0  # 0 means inherit

    def test_configure_with_debug_level(self) -> None:
        """Should configure logging with DEBUG level."""
        LoggerFactory.configure(level=logging.DEBUG)
        logger = LoggerFactory.get_logger("test.debug")
        assert logger.level == logging.DEBUG or logger.level == 0

    def test_get_logger_returns_consistent_instance(self) -> None:
        """Should return same logger instance for same name."""
        LoggerFactory.configure()
        logger1 = LoggerFactory.get_logger("consistent")
        logger2 = LoggerFactory.get_logger("consistent")
        assert logger1 is logger2

    def test_configure_with_custom_handler(self) -> None:
        """Should use custom handlers when provided."""
        handler = logging.StreamHandler()
        LoggerFactory.configure(handlers=[handler])
        logger = LoggerFactory.get_logger("custom")
        assert len(logger.handlers) > 0

    def test_logger_has_handlers_after_configure(self) -> None:
        """Logger should have handlers after configuration."""
        LoggerFactory.configure()
        logger = LoggerFactory.get_logger("with_handlers")
        assert len(logger.handlers) > 0

    def test_get_logger_without_configure(self) -> None:
        """Should auto-configure if not already configured."""
        logger = LoggerFactory.get_logger("auto_config")
        assert logger is not None
        assert logger.level != 0 or len(logger.handlers) > 0

    def test_logger_propagate_is_false(self) -> None:
        """Logger propagate should be False to avoid duplicate logs."""
        LoggerFactory.configure()
        logger = LoggerFactory.get_logger("no_propagate")
        assert logger.propagate is False

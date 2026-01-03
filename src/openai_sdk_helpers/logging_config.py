"""Centralized logging configuration for openai-sdk-helpers.

Provides a centralized factory for creating and configuring loggers
with consistent formatting and handler management.
"""

import logging
import threading
from typing import Any


class LoggerFactory:
    """Centralized logger creation and configuration.

    Manages logger initialization and configuration to ensure consistent
    logging behavior across the entire SDK. Thread-safe.

    Examples
    --------
    Configure logging once at application startup:

    >>> from openai_sdk_helpers.logging_config import LoggerFactory
    >>> import logging
    >>> LoggerFactory.configure(
    ...     level=logging.DEBUG,
    ...     handlers=[logging.StreamHandler()],
    ... )

    Get a logger instance in your module:

    >>> logger = LoggerFactory.get_logger("openai_sdk_helpers.agent")
    >>> logger.debug("Debug message")
    """

    _initialized = False
    _log_level = logging.INFO
    _handlers: list[logging.Handler] = []
    _lock = threading.Lock()

    @classmethod
    def configure(
        cls,
        level: int = logging.INFO,
        handlers: list[logging.Handler] | None = None,
    ) -> None:
        """Configure logging globally.

        Parameters
        ----------
        level : int
            Logging level (e.g., logging.DEBUG, logging.INFO).
            Default is logging.INFO.
        handlers : list[logging.Handler] | None
            List of logging handlers. If None, a default
            StreamHandler is created. Default is None.

        Notes
        -----
        This method is thread-safe and can be called multiple times.
        """
        with cls._lock:
            cls._log_level = level
            if handlers:
                cls._handlers = handlers
            else:
                handler = logging.StreamHandler()
                handler.setLevel(level)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                cls._handlers = [handler]
            cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get configured logger instance.

        Parameters
        ----------
        name : str
            Logger name, typically __name__ of calling module.

        Returns
        -------
        logging.Logger
            Configured logger instance.
        """
        logger = logging.getLogger(name)

        # Skip configuration if already configured
        if logger.handlers:
            return logger

        with cls._lock:
            if not cls._initialized:
                cls.configure()

            for handler in cls._handlers:
                logger.addHandler(handler)

            logger.setLevel(cls._log_level)
            logger.propagate = False

        return logger

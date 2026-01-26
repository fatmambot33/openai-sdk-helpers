"""Centralized logging configuration for openai-sdk-helpers."""

import logging


def log(
    message: str,
    level: int = logging.INFO,
    *,
    logger_name: str = "openai_sdk_helpers",
    exc: BaseException | None = None,
) -> None:
    """Log a message using Python's standard logging.

    Parameters
    ----------
    message : str
        The message to log.
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
        Default is logging.INFO.
    logger_name : str
        Name of the logger. Default is "openai_sdk_helpers".
    exc : BaseException or None, optional
        Exception instance to include with the log record. Default is None.

    Returns
    -------
    None
        Return None after emitting the log entry.

    Examples
    --------
    >>> from openai_sdk_helpers.logging_config import log
    >>> log("Operation completed")
    >>> log("Debug info", level=logging.DEBUG)
    """
    logger = logging.getLogger(logger_name)
    exc_info = None
    if exc is not None:
        exc_info = (type(exc), exc, exc.__traceback__)
    logger.log(level, message, exc_info=exc_info)


__all__ = ["log"]

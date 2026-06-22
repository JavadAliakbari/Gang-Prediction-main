"""Centralized logging configuration for AMLGentex.

This module provides a unified logging system to replace mixed print() + logging
approaches throughout the codebase.
"""
import logging
import sys
from typing import Optional

# Package-level loggers
_PACKAGE_LOGGERS = [
    'src.data_creation',
    'src.feature_engineering',
    'src.data_tuning',
    'src.ml',
]


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent formatting.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def set_verbosity(verbose: bool = True):
    """Set logging level for all AMLGentex modules.

    Args:
        verbose: If True, show INFO messages. If False, only WARNING+.
    """
    level = logging.INFO if verbose else logging.WARNING

    # Update root logger level
    logging.getLogger().setLevel(level)

    # Update all handlers
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)

    # Update package loggers
    for pkg in _PACKAGE_LOGGERS:
        logging.getLogger(pkg).setLevel(level)


def configure_logging(verbose: bool = True, log_file: Optional[str] = None):
    """Configure logging for the entire package.

    Call once at application startup (e.g., in scripts).

    Args:
        verbose: If True, show INFO messages. If False, only WARNING+.
        log_file: Optional path to log file. If provided, logs to both file and stdout.
    """
    level = logging.INFO if verbose else logging.WARNING

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(levelname)s - %(name)s - %(message)s',
        handlers=handlers,
        force=True
    )
    set_verbosity(verbose)

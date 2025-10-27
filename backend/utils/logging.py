"""Logging utilities."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days",
    format: Optional[str] = None,
) -> None:
    """Setup loguru logger with file and console handlers.

    Args:
        log_file: Path to log file (optional)
        level: Logging level
        rotation: When to rotate log file
        retention: How long to keep old logs
        format: Custom log format string
    """
    # Remove default handler
    logger.remove()

    # Default format
    if format is None:
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logger initialized with level: {level}")


def get_logger(name: str):
    """Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


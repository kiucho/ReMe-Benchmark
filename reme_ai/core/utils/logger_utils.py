"""Logging configuration module for application-wide tracing."""

import os
import sys
from datetime import datetime


def init_logger(log_dir: str = "logs", level: str = "INFO") -> None:
    """Initialize the logger with both file and console handlers."""
    from loguru import logger

    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate filename based on the current timestamp
    current_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{current_ts}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Configure file-based logging with rotation and compression
    logger.add(
        log_filepath,
        level=level,
        rotation="00:00",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )

    # Configure colorized standard output logging
    logger.add(
        sink=sys.stdout,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        colorize=True,
    )

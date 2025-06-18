# ===============================
# 1. cadgen_project/utils/logging_config.py
# ===============================

import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from cadgen_project.config import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward loguru sinks."""

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(log_level: str = None, log_file: str = None) -> None:
    """
    Setup logging configuration with both console and file outputs.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Remove default loguru handler
    logger.remove()

    # Use settings if parameters not provided
    log_level = log_level or settings.log_level

    # Console handler with rich formatting
    console = Console()

    # Add console handler
    logger.add(
        sink=sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if specified or if not in debug mode
    if log_file or not settings.debug:
        log_file = log_file or f"logs/cadgen_{settings.log_level.lower()}.log"
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            sink=log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logger.info(f"Logging initialized with level: {log_level}")


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)

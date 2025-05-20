"""
Logging utilities for alpha_recall.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default log configuration
DEFAULT_LOG_FILE = str(Path.home() / "Library/Logs/Alpha/alpha-recall.log")
DEFAULT_LOG_LEVEL = "INFO"

# Get log configuration from environment variables
LOG_FILE = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)
LOG_LEVEL = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()


def ensure_log_path_exists(log_file_path: str) -> Path:
    """
    Ensure the parent directory for the log file exists.

    Args:
        log_file_path: Path to the log file

    Returns:
        Path object for the log file
    """
    # Expand user directory if path contains ~
    log_path = Path(log_file_path).expanduser()

    # Create parent directory if it doesn't exist
    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_path


def configure_logging(
    log_file: Optional[str] = None, log_level: Optional[str] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        log_file: Optional override for log file path
        log_level: Optional override for log level
    """
    # Use provided values or fall back to environment variables
    log_path = ensure_log_path_exists(log_file or LOG_FILE)
    level_name = (log_level or LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),  # Also log to console
        ],
    )

    # Log initialization
    logger = logging.getLogger("alpha_recall")
    logger.info(f"Logging initialized at {log_path} with level {level_name}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Name for the logger

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"alpha_recall.{name}")

"""Structured logging configuration for Alpha-Recall."""

import logging
import sys
import warnings

import structlog
from rich.console import Console
from rich.json import JSON

from .config import settings


class RichJSONRenderer:
    """Custom structlog renderer that uses Rich for JSON pretty-printing."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console(file=sys.stdout, stderr=False)

    def __call__(self, logger, method_name, event_dict):
        # Convert the event dict to JSON and pretty-print with Rich
        rich_json = JSON.from_data(event_dict)
        self.console.print(rich_json)
        return ""  # Return empty string since Rich handles the output


def configure_logging() -> structlog.stdlib.BoundLogger:
    """Configure structured logging based on settings."""

    # Suppress known deprecation warnings from dependencies
    warnings.filterwarnings("ignore", message="websockets.legacy is deprecated.*")
    warnings.filterwarnings(
        "ignore", message="websockets.server.WebSocketServerProtocol is deprecated.*"
    )

    # Configure standard library logging first
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        handlers=[],  # We'll add handlers below
    )

    # Shared processors for both formats
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if settings.log_format == "json":
        # Production JSON format
        processors = shared_processors + [
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]

        # Use standard StreamHandler
        handler = logging.StreamHandler(sys.stdout)

    elif settings.log_format == "rich_json":
        # Rich JSON format - structured data with beautiful syntax highlighting
        processors = shared_processors + [
            structlog.processors.TimeStamper(fmt="iso"),
            RichJSONRenderer(),
        ]

        # Use standard StreamHandler for console output
        handler = logging.StreamHandler(sys.stdout)

    else:  # rich format - always use colors for rich format
        # Use ConsoleRenderer for beautiful human-readable output
        processors = shared_processors + [
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

        # Use standard StreamHandler for console output
        handler = logging.StreamHandler(sys.stdout)

    # Add the handler to root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Return a logger for immediate use
    return structlog.get_logger("alpha_recall")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given name."""
    return structlog.get_logger(name)


# Module-level logger for convenience
logger = structlog.get_logger(__name__)

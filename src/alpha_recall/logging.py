"""Structured logging configuration for Alpha-Recall."""

import logging
import sys
import warnings

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


def configure_logging() -> structlog.stdlib.BoundLogger:
    """Configure structured logging based on settings."""
    
    # Suppress known deprecation warnings from dependencies
    warnings.filterwarnings("ignore", message="websockets.legacy is deprecated.*")
    warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated.*")
    
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
            structlog.processors.JSONRenderer()
        ]
        
        # Use standard StreamHandler
        handler = logging.StreamHandler(sys.stdout)
        
    else:  # rich format - always use colors for rich format
        # Use RichHandler approach - let Rich handle all formatting
        processors = shared_processors + [
            # Don't use ConsoleRenderer with RichHandler - just pass structured data
            structlog.processors.KeyValueRenderer(key_order=["level", "event"])
        ]
        
        # Use RichHandler for beautiful output, force terminal for Docker
        console = Console(file=sys.stdout, force_terminal=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
    
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
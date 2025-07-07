"""Correlation ID utilities for request tracing across Alpha-Recall."""

import uuid
from contextvars import ContextVar

import structlog

# Context variable to store the current correlation ID
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def generate_correlation_id(prefix: str = "req") -> str:
    """Generate a new correlation ID with optional prefix.

    Args:
        prefix: Optional prefix for the correlation ID (default: "req")

    Returns:
        A new correlation ID in format: prefix_xxxxxxxx
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set
    """
    _correlation_id.set(correlation_id)

    # Also set it in structlog's context variables for automatic inclusion
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context.

    Returns:
        The current correlation ID or None if not set
    """
    return _correlation_id.get()


def create_child_correlation_id(operation: str) -> str:
    """Create a child correlation ID for sub-operations.

    Args:
        operation: Name of the child operation

    Returns:
        A child correlation ID in format: parent_correlation_id.operation
    """
    parent_id = get_correlation_id()
    if parent_id:
        return f"{parent_id}.{operation}"
    else:
        # If no parent, create a new root ID
        return generate_correlation_id(operation)


def with_correlation_id(correlation_id: str | None = None, prefix: str = "req"):
    """Decorator to automatically set correlation ID for a function.

    Args:
        correlation_id: Specific correlation ID to use, or None to generate
        prefix: Prefix for generated correlation ID if correlation_id is None

    Usage:
        @with_correlation_id()
        def my_function():
            # correlation ID is automatically set
            pass

        @with_correlation_id("custom_id")
        def my_function():
            # uses "custom_id" as correlation ID
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate or use provided correlation ID
            corr_id = correlation_id or generate_correlation_id(prefix)

            # Set correlation ID in context
            token = _correlation_id.set(corr_id)
            structlog.contextvars.bind_contextvars(correlation_id=corr_id)

            try:
                return func(*args, **kwargs)
            finally:
                # Clean up context
                _correlation_id.reset(token)
                structlog.contextvars.clear_contextvars()

        return wrapper

    return decorator


def with_async_correlation_id(correlation_id: str | None = None, prefix: str = "req"):
    """Async decorator to automatically set correlation ID for an async function.

    Args:
        correlation_id: Specific correlation ID to use, or None to generate
        prefix: Prefix for generated correlation ID if correlation_id is None
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate or use provided correlation ID
            corr_id = correlation_id or generate_correlation_id(prefix)

            # Set correlation ID in context
            token = _correlation_id.set(corr_id)
            structlog.contextvars.bind_contextvars(correlation_id=corr_id)

            try:
                return await func(*args, **kwargs)
            finally:
                # Clean up context
                _correlation_id.reset(token)
                structlog.contextvars.clear_contextvars()

        return wrapper

    return decorator

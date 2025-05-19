"""
Retry utilities for error handling and recovery.

This module provides decorators and utilities for handling retries on
transient errors, allowing for more robust connections to external services.
"""

import asyncio
import functools
import time
from typing import Any, Callable, List, Optional, Type, TypeVar, Union, cast

from alpha_recall.logging_utils import get_logger

# Configure logging
logger = get_logger(__name__)

T = TypeVar('T')

def async_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    exceptions_to_retry: Optional[List[Type[Exception]]] = None,
    error_messages_to_retry: Optional[List[str]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying asynchronous functions when exceptions occur.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for increasing delay between retries
        max_delay: Maximum delay between retries in seconds
        exceptions_to_retry: List of exception types to retry on (defaults to Exception)
        error_messages_to_retry: List of error message substrings to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            exceptions_to_catch = exceptions_to_retry or [Exception]
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries + 1):  # +1 for the initial attempt
                try:
                    # Perform the operation
                    return await func(*args, **kwargs)
                    
                except tuple(exceptions_to_catch) as e:
                    last_exception = e
                    
                    # Check if this exception should be retried based on message
                    should_retry = True
                    if error_messages_to_retry:
                        should_retry = any(msg in str(e) for msg in error_messages_to_retry)
                    
                    if not should_retry:
                        # If this exception type doesn't match our retry messages, re-raise
                        raise
                    
                    # If this was the last attempt, re-raise the exception
                    if attempt >= max_retries:
                        logger.warning(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts: {str(e)}"
                        )
                        raise
                    
                    # Log the retry attempt
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after error: {str(e)}. Retrying in {delay:.2f}s"
                    )
                    
                    # Wait before the next attempt with exponential backoff
                    await asyncio.sleep(delay)
                    
                    # Update delay with backoff, but cap at max_delay
                    delay = min(delay * backoff_factor, max_delay)
            
            # This should never be reached due to the re-raise in the loop,
            # but we include it for completeness
            if last_exception:
                raise last_exception
            return None
            
        return cast(Callable[..., T], wrapper)
    return decorator

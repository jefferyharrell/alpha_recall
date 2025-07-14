"""Base Redis service with shared connection logic."""

import time

import redis

from ..config import settings
from ..logging import get_logger
from ..utils.correlation import get_correlation_id

logger = get_logger("services.redis_base")


class RedisBaseService:
    """Base service for Redis operations with shared connection logic."""

    def __init__(self):
        """Initialize the Redis base service."""
        self._client: redis.Redis | None = None
        self._connection_tested = False

    @property
    def client(self) -> redis.Redis:
        """Get or create Redis client with connection pooling."""
        if self._client is None:
            # Use connection pooling
            pool = redis.ConnectionPool.from_url(
                settings.redis_uri, max_connections=10, decode_responses=False
            )
            self._client = redis.Redis(connection_pool=pool)
            logger.debug(
                "Created Redis client with connection pool", uri=settings.redis_uri
            )
        return self._client

    def test_connection(self) -> bool:
        """Test the Redis connection."""
        if self._connection_tested:
            return True

        try:
            start_time = time.perf_counter()
            # Simple ping test
            result = self.client.ping()
            test_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            if result:
                self._connection_tested = True
                logger.info(
                    "Redis connection test successful",
                    test_time_ms=test_time_ms,
                    correlation_id=get_correlation_id(),
                )
                return True
            else:
                logger.error(
                    "Redis connection test failed - ping returned False",
                    correlation_id=get_correlation_id(),
                )
                return False
        except Exception as e:
            logger.error(
                "Redis connection test failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            return False

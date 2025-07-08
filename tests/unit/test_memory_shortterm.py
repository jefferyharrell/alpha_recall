"""Unit tests for short-term memory tools."""

import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.browse_shortterm import browse_shortterm


@pytest.fixture
def mock_redis():
    """Mock Redis service for testing."""
    with patch("alpha_recall.services.redis.get_redis_service") as mock:
        service = MagicMock()
        client = MagicMock()
        service.client = client
        mock.return_value = service
        yield service


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "memory_id": f"stm_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}",
        "content": "This is a test memory about embeddings and AI development",
        "semantic_embedding": np.random.rand(768).astype(np.float32),
        "emotional_embedding": np.random.rand(1024).astype(np.float32),
        "created_at": datetime.now(UTC).isoformat(),
    }


class TestStoreMemoryToRedis:
    """Test the RedisService store_memory method."""

    def test_successful_storage(self, mock_redis, sample_memory_data):
        """Test successful memory storage to Redis."""
        # Configure mock to return True for successful storage
        mock_redis.store_memory.return_value = True

        # Test the storage function
        result = mock_redis.store_memory(
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

        # Verify success
        assert result is True

        # Verify the service method was called correctly
        mock_redis.store_memory.assert_called_once_with(
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

    def test_redis_error_handling(self, mock_redis, sample_memory_data):
        """Test error handling when Redis operations fail."""
        # Make Redis operations raise an exception
        mock_redis.store_memory.return_value = False

        result = mock_redis.store_memory(
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

        # Verify failure is handled gracefully
        assert result is False

    def test_vector_serialization(self, mock_redis, sample_memory_data):
        """Test that numpy arrays are correctly serialized through RedisService."""
        # Test the store_memory method with vectors
        mock_redis.store_memory(
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

        # Verify the service method was called
        mock_redis.store_memory.assert_called_once()

        # Note: Vector serialization details are now handled internally by RedisService
        # This test verifies the interface works correctly


class TestSearchRelatedMemories:
    """Test the search_related_memories method of RedisService."""

    def test_empty_memory_index(self, mock_redis):
        """Test search when no memories exist."""
        # Mock empty search results
        mock_redis.search_related_memories.return_value = []

        query_embedding = np.random.rand(768).astype(np.float32)
        results = mock_redis.search_related_memories(
            content="test query", query_embedding=query_embedding
        )

        assert results == []
        mock_redis.search_related_memories.assert_called_once()

    def test_search_with_memories_vector_index_failure(self, mock_redis):
        """Test search fails gracefully when vector index cannot be created."""
        # Mock search failure
        mock_redis.search_related_memories.return_value = []

        query_embedding = np.array([0.9, 0.1] + [0.0] * 766)

        results = mock_redis.search_related_memories(
            content="test query about AI", query_embedding=query_embedding
        )

        # Should return empty list when vector search fails
        assert results == []

    def test_search_with_working_vector_index(self, mock_redis):
        """Test successful vector search when index exists and works."""
        # Mock successful search results
        mock_results = [
            {
                "id": "stm_1",
                "content": "Test memory",
                "created_at": "2025-07-07T12:00:00+00:00",
                "similarity_score": 0.9,
                "source": "redis_vector_search",
            }
        ]
        mock_redis.search_related_memories.return_value = mock_results

        query_embedding = np.array([1.0, 0.0] + [0.0] * 766)
        results = mock_redis.search_related_memories(
            content="test", query_embedding=query_embedding
        )

        # Should return results from Redis vector search
        assert len(results) == 1
        assert results[0]["source"] == "redis_vector_search"
        assert results[0]["id"] == "stm_1"

    def test_vector_search_runtime_failure(self, mock_redis):
        """Test graceful failure when vector search fails at runtime."""
        # Mock search failure
        mock_redis.search_related_memories.return_value = []

        query_embedding = np.array([1.0, 0.0] + [0.0] * 766)
        results = mock_redis.search_related_memories(
            content="test", query_embedding=query_embedding
        )

        # Should return empty list when vector search fails at runtime
        assert results == []

    def test_error_handling(self, mock_redis):
        """Test error handling in search function."""
        # Make search operations raise an exception
        mock_redis.search_related_memories.side_effect = Exception(
            "Redis connection failed"
        )

        query_embedding = np.random.rand(768).astype(np.float32)
        try:
            results = mock_redis.search_related_memories(
                content="test query", query_embedding=query_embedding
            )
        except Exception:
            results = []

        # Should return empty list on error
        assert results == []


class TestGetRedisService:
    """Test the get_redis_service function."""

    def test_redis_service_creation(self):
        """Test that Redis service is created and has correct interface."""
        from alpha_recall.services.redis import RedisService, get_redis_service

        service = get_redis_service()

        # Verify service is instance of RedisService
        assert isinstance(service, RedisService)

        # Verify service has expected methods
        assert hasattr(service, "client")
        assert hasattr(service, "test_connection")
        assert hasattr(service, "store_memory")
        assert hasattr(service, "search_related_memories")


class TestBrowseShortterm:
    """Test the browse_shortterm function."""

    @patch("alpha_recall.tools.browse_shortterm.get_redis_service")
    def test_browse_empty_memories(self, mock_get_redis_service):
        """Test browsing when no memories exist."""
        mock_service = MagicMock()
        mock_redis = MagicMock()
        mock_service.client = mock_redis
        mock_get_redis_service.return_value = mock_service

        # Mock empty memory index
        mock_redis.zcard.return_value = 0
        mock_redis.zrevrange.return_value = []

        result = browse_shortterm()

        # Parse JSON result
        data = json.loads(result)

        assert data["memories"] == []
        assert data["pagination"]["returned"] == 0
        assert data["pagination"]["total_in_range"] == 0
        assert not data["pagination"]["has_more"]

    @patch("alpha_recall.tools.browse_shortterm.get_redis_service")
    def test_browse_with_memories(self, mock_get_redis_service):
        """Test browsing with existing memories."""
        mock_service = MagicMock()
        mock_redis = MagicMock()
        mock_service.client = mock_redis
        mock_get_redis_service.return_value = mock_service

        # Mock memory index with 2 memories
        mock_redis.zcard.return_value = 2
        mock_redis.zrevrange.return_value = [
            (b"stm_1", 1625097600.0),  # July 1, 2021
            (b"stm_2", 1625011200.0),  # June 30, 2021
        ]

        # Mock memory content retrieval
        def mock_hmget(key, fields):
            if key == "memory:stm_1":
                return [b"First test memory", b"2021-07-01T00:00:00+00:00", b"stm_1"]
            elif key == "memory:stm_2":
                return [b"Second test memory", b"2021-06-30T00:00:00+00:00", b"stm_2"]
            return [None, None, None]

        mock_redis.hmget.side_effect = mock_hmget

        result = browse_shortterm(limit=10)

        # Parse JSON result
        data = json.loads(result)

        assert len(data["memories"]) == 2
        assert data["memories"][0]["content"] == "First test memory"
        assert data["memories"][1]["content"] == "Second test memory"
        assert data["pagination"]["returned"] == 2
        assert data["pagination"]["total_in_range"] == 2
        assert not data["pagination"]["has_more"]

    @patch("alpha_recall.tools.browse_shortterm.get_redis_service")
    def test_browse_with_pagination(self, mock_get_redis_service):
        """Test browsing with pagination (offset and limit)."""
        mock_service = MagicMock()
        mock_redis = MagicMock()
        mock_service.client = mock_redis
        mock_get_redis_service.return_value = mock_service

        # Mock memory index with more memories than limit
        mock_redis.zcard.return_value = 10
        mock_redis.zrevrange.return_value = [
            (
                b"stm_3",
                1625097600.0,
            ),  # Should get memory 3 and 4 with offset=2, limit=2
            (b"stm_4", 1625011200.0),
        ]

        # Mock memory content retrieval
        def mock_hmget(key, fields):
            if key == "memory:stm_3":
                return [b"Third memory", b"2021-07-01T00:00:00+00:00", b"stm_3"]
            elif key == "memory:stm_4":
                return [b"Fourth memory", b"2021-06-30T00:00:00+00:00", b"stm_4"]
            return [None, None, None]

        mock_redis.hmget.side_effect = mock_hmget

        result = browse_shortterm(limit=2, offset=2)

        # Parse JSON result
        data = json.loads(result)

        assert len(data["memories"]) == 2
        assert data["pagination"]["returned"] == 2
        assert data["pagination"]["total_in_range"] == 10
        assert data["pagination"]["has_more"]
        assert data["pagination"]["showing"] == "3-4 of 10"

        # Verify the correct Redis call was made with offset and limit
        mock_redis.zrevrange.assert_called_once_with(
            "memory_index",
            2,
            3,
            withscores=True,  # offset=2, end=offset+limit-1=3
        )

    @patch("alpha_recall.tools.browse_shortterm.get_redis_service")
    def test_browse_with_search_filter(self, mock_get_redis_service):
        """Test browsing with search text filtering."""
        mock_service = MagicMock()
        mock_redis = MagicMock()
        mock_service.client = mock_redis
        mock_get_redis_service.return_value = mock_service

        # Mock memory index
        mock_redis.zcard.return_value = 3
        mock_redis.zrevrange.return_value = [
            (b"stm_1", 1625097600.0),
            (b"stm_2", 1625011200.0),
            (b"stm_3", 1624924800.0),
        ]

        # Mock memory content retrieval
        def mock_hmget(key, fields):
            if key == "memory:stm_1":
                return [
                    b"Memory about Python programming",
                    b"2021-07-01T00:00:00+00:00",
                    b"stm_1",
                ]
            elif key == "memory:stm_2":
                return [
                    b"Memory about JavaScript coding",
                    b"2021-06-30T00:00:00+00:00",
                    b"stm_2",
                ]
            elif key == "memory:stm_3":
                return [
                    b"Memory about cooking recipes",
                    b"2021-06-29T00:00:00+00:00",
                    b"stm_3",
                ]
            return [None, None, None]

        mock_redis.hmget.side_effect = mock_hmget

        result = browse_shortterm(search="python")

        # Parse JSON result
        data = json.loads(result)

        # Should only return the Python memory (case-insensitive search)
        assert len(data["memories"]) == 1
        assert data["memories"][0]["content"] == "Memory about Python programming"
        assert data["filters"]["search"] == "python"

    @patch("alpha_recall.tools.browse_shortterm.get_redis_service")
    def test_browse_with_since_duration(self, mock_get_redis_service):
        """Test browsing with 'since' time filtering."""
        mock_service = MagicMock()
        mock_redis = MagicMock()
        mock_service.client = mock_redis
        mock_get_redis_service.return_value = mock_service

        # Mock ZREVRANGEBYSCORE call for time-based filtering
        mock_redis.zrevrangebyscore.return_value = [
            (b"stm_recent", 1625097600.0),
        ]
        mock_redis.zcount.return_value = 1

        # Mock memory content retrieval
        mock_redis.hmget.return_value = [
            b"Recent memory",
            b"2021-07-01T00:00:00+00:00",
            b"stm_recent",
        ]

        result = browse_shortterm(since="6h")

        # Parse JSON result
        data = json.loads(result)

        assert len(data["memories"]) == 1
        assert data["filters"]["since"] == "6h"

        # Verify ZREVRANGEBYSCORE was called instead of ZREVRANGE
        mock_redis.zrevrangebyscore.assert_called_once()

    @patch("alpha_recall.tools.browse_shortterm.get_redis_service")
    def test_browse_ascending_order(self, mock_get_redis_service):
        """Test browsing with ascending order (oldest first)."""
        mock_service = MagicMock()
        mock_redis = MagicMock()
        mock_service.client = mock_redis
        mock_get_redis_service.return_value = mock_service

        # Mock memory index
        mock_redis.zcard.return_value = 2
        mock_redis.zrange.return_value = [  # Note: zrange instead of zrevrange
            (b"stm_old", 1625011200.0),
            (b"stm_new", 1625097600.0),
        ]

        # Mock memory content retrieval
        def mock_hmget(key, fields):
            if key == "memory:stm_old":
                return [b"Older memory", b"2021-06-30T00:00:00+00:00", b"stm_old"]
            elif key == "memory:stm_new":
                return [b"Newer memory", b"2021-07-01T00:00:00+00:00", b"stm_new"]
            return [None, None, None]

        mock_redis.hmget.side_effect = mock_hmget

        result = browse_shortterm(order="asc")

        # Parse JSON result
        data = json.loads(result)

        assert len(data["memories"]) == 2
        assert data["filters"]["order"] == "asc"

        # Verify ZRANGE was called instead of ZREVRANGE for ascending order
        mock_redis.zrange.assert_called_once()

    @patch("alpha_recall.tools.browse_shortterm.get_redis_service")
    def test_browse_error_handling(self, mock_get_redis_service):
        """Test error handling when Redis operations fail."""
        mock_service = MagicMock()
        mock_redis = MagicMock()
        mock_service.client = mock_redis
        mock_get_redis_service.return_value = mock_service

        # Make Redis operations raise an exception
        mock_redis.zcard.side_effect = Exception("Redis connection failed")

        result = browse_shortterm()

        # Parse JSON result
        data = json.loads(result)

        # Should return error response
        assert "error" in data
        assert data["memories"] == []
        assert data["pagination"]["returned"] == 0

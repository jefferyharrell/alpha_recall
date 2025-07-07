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

from alpha_recall.tools.memory_shortterm import (
    get_redis_client,
    search_related_memories,
    store_memory_to_redis,
)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch("alpha_recall.tools.memory_shortterm.get_redis_client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


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
    """Test the store_memory_to_redis function."""

    def test_successful_storage(self, mock_redis, sample_memory_data):
        """Test successful memory storage to Redis."""
        # Test the storage function
        result = store_memory_to_redis(
            client=mock_redis,
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

        # Verify success
        assert result is True

        # Verify Redis calls were made correctly
        memory_key = f"memory:{sample_memory_data['memory_id']}"

        # Check that hset was called with correct data
        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == memory_key  # First positional arg is the key

        # Check the mapping argument
        mapping = call_args[1]["mapping"]
        assert mapping["content"] == sample_memory_data["content"]
        assert mapping["created_at"] == sample_memory_data["created_at"]
        assert mapping["id"] == sample_memory_data["memory_id"]

        # Check that vectors were stored correctly
        # Semantic vector should be binary for Redis vector search
        assert isinstance(mapping["semantic_vector"], bytes)
        # Emotional vector stored as JSON
        stored_emotional = json.loads(mapping["emotional_vector"])
        assert len(stored_emotional) == 1024

        # Verify binary semantic vector has correct size (768 floats * 4 bytes each)
        assert len(mapping["semantic_vector"]) == 768 * 4

        # Check that expire was called for the memory
        mock_redis.expire.assert_any_call(
            memory_key, mock_redis.expire.call_args_list[0][0][1]
        )

        # Check that zadd was called for the index
        mock_redis.zadd.assert_called_once_with(
            "memory_index",
            {
                sample_memory_data["memory_id"]: mock_redis.zadd.call_args[0][1][
                    sample_memory_data["memory_id"]
                ]
            },
        )

    def test_redis_error_handling(self, mock_redis, sample_memory_data):
        """Test error handling when Redis operations fail."""
        # Make Redis operations raise an exception
        mock_redis.hset.side_effect = Exception("Redis connection failed")

        result = store_memory_to_redis(
            client=mock_redis,
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

        # Verify failure is handled gracefully
        assert result is False

    def test_vector_serialization(self, mock_redis, sample_memory_data):
        """Test that numpy arrays are correctly serialized to binary and JSON."""
        store_memory_to_redis(
            client=mock_redis,
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

        # Get the mapping that was passed to hset
        mapping = mock_redis.hset.call_args[1]["mapping"]

        # Verify binary semantic vector can be round-tripped
        semantic_binary = mapping["semantic_vector"]
        semantic_roundtrip = np.frombuffer(semantic_binary, dtype=np.float32)
        np.testing.assert_array_almost_equal(
            semantic_roundtrip, sample_memory_data["semantic_embedding"]
        )

        # Verify JSON emotional vector can be round-tripped
        emotional_roundtrip = np.array(json.loads(mapping["emotional_vector"]))

        np.testing.assert_array_almost_equal(
            emotional_roundtrip, sample_memory_data["emotional_embedding"]
        )


class TestSearchRelatedMemories:
    """Test the search_related_memories function."""

    def test_empty_memory_index(self, mock_redis):
        """Test search when no memories exist."""
        # Mock empty memory index
        mock_redis.zcard.return_value = 0  # No memories in index

        query_embedding = np.random.rand(768).astype(np.float32)
        results = search_related_memories(mock_redis, "test query", query_embedding)

        assert results == []
        mock_redis.zcard.assert_called_once_with("memory_index")

    def test_search_with_memories_vector_index_failure(self, mock_redis):
        """Test search fails gracefully when vector index cannot be created."""
        # Mock that we have memories but vector search fails (no fallback)
        mock_redis.zcard.return_value = 3  # 3 memories exist

        # Mock FT.INFO to fail (index doesn't exist) and FT.CREATE to also fail
        def mock_execute_command(cmd, *args):
            if cmd == "FT.INFO":
                from redis import ResponseError

                raise ResponseError("Unknown index name")
            elif cmd == "FT.CREATE":
                raise Exception("Could not create index")
            raise Exception("Unexpected command")

        mock_redis.execute_command.side_effect = mock_execute_command

        query_embedding = np.array([0.9, 0.1] + [0.0] * 766)

        results = search_related_memories(
            mock_redis, "test query about AI", query_embedding
        )

        # Should return empty list when vector index setup fails
        assert results == []

    def test_search_with_working_vector_index(self, mock_redis):
        """Test successful vector search when index exists and works."""
        mock_redis.zcard.return_value = 2

        # Mock successful vector index check and search
        def mock_execute_command(cmd, *args):
            if cmd == "FT.INFO":
                return "index info response"
            elif cmd == "FT.SEARCH":
                # Mock successful search result with 1 result
                return [
                    1,  # Total results
                    b"memory:stm_1",  # Document key
                    [
                        b"content",
                        b"Test memory",
                        b"created_at",
                        b"2025-07-07T12:00:00+00:00",
                        b"id",
                        b"stm_1",
                        b"similarity_score",
                        b"0.1",
                    ],  # Field-value pairs
                ]
            raise Exception("Unexpected command")

        mock_redis.execute_command.side_effect = mock_execute_command

        query_embedding = np.array([1.0, 0.0] + [0.0] * 766)
        results = search_related_memories(mock_redis, "test", query_embedding)

        # Should return results from Redis vector search
        assert len(results) == 1
        assert results[0]["source"] == "redis_vector_search"
        assert results[0]["id"] == "stm_1"

    def test_vector_search_runtime_failure(self, mock_redis):
        """Test graceful failure when vector search fails at runtime."""
        mock_redis.zcard.return_value = 1

        # Mock index exists but search fails
        def mock_execute_command(cmd, *args):
            if cmd == "FT.INFO":
                return "index exists"
            elif cmd == "FT.SEARCH":
                raise Exception("Search failed")
            raise Exception("Unexpected command")

        mock_redis.execute_command.side_effect = mock_execute_command

        query_embedding = np.array([1.0, 0.0] + [0.0] * 766)
        results = search_related_memories(mock_redis, "test", query_embedding)

        # Should return empty list when vector search fails at runtime
        assert results == []

    def test_error_handling(self, mock_redis):
        """Test error handling in search function."""
        # Make Redis operations raise an exception early
        mock_redis.zcard.side_effect = Exception("Redis connection failed")

        query_embedding = np.random.rand(768).astype(np.float32)
        results = search_related_memories(mock_redis, "test query", query_embedding)

        # Should return empty list on error
        assert results == []


class TestGetRedisClient:
    """Test the get_redis_client function."""

    @patch("alpha_recall.tools.memory_shortterm.redis.from_url")
    @patch("alpha_recall.tools.memory_shortterm.settings")
    def test_redis_client_creation(self, mock_settings, mock_redis_from_url):
        """Test that Redis client is created with correct URI."""
        mock_settings.redis_uri = "redis://localhost:6379/0"

        get_redis_client()

        mock_redis_from_url.assert_called_once_with("redis://localhost:6379/0")

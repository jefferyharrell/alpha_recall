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

        # Check that vectors were stored as JSON
        stored_semantic = json.loads(mapping["semantic_vector"])
        stored_emotional = json.loads(mapping["emotional_vector"])
        assert len(stored_semantic) == 768
        assert len(stored_emotional) == 1024

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
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

        # Verify failure is handled gracefully
        assert result is False

    def test_vector_serialization(self, mock_redis, sample_memory_data):
        """Test that numpy arrays are correctly serialized to JSON."""
        store_memory_to_redis(
            memory_id=sample_memory_data["memory_id"],
            content=sample_memory_data["content"],
            semantic_embedding=sample_memory_data["semantic_embedding"],
            emotional_embedding=sample_memory_data["emotional_embedding"],
            created_at=sample_memory_data["created_at"],
        )

        # Get the mapping that was passed to hset
        mapping = mock_redis.hset.call_args[1]["mapping"]

        # Verify vectors can be round-tripped through JSON
        semantic_roundtrip = np.array(json.loads(mapping["semantic_vector"]))
        emotional_roundtrip = np.array(json.loads(mapping["emotional_vector"]))

        np.testing.assert_array_almost_equal(
            semantic_roundtrip, sample_memory_data["semantic_embedding"]
        )
        np.testing.assert_array_almost_equal(
            emotional_roundtrip, sample_memory_data["emotional_embedding"]
        )


class TestSearchRelatedMemories:
    """Test the search_related_memories function."""

    def test_empty_memory_index(self, mock_redis):
        """Test search when no memories exist."""
        # Mock empty memory index
        mock_redis.zrevrange.return_value = []

        query_embedding = np.random.rand(768).astype(np.float32)
        results = search_related_memories("test query", query_embedding)

        assert results == []
        mock_redis.zrevrange.assert_called_once_with("memory_index", 0, -1)

    def test_search_with_memories(self, mock_redis):
        """Test search with existing memories."""
        # Mock memory IDs in index
        memory_ids = [b"stm_1", b"stm_2", b"stm_3"]
        mock_redis.zrevrange.return_value = memory_ids

        # Mock memory data
        similar_embedding = np.array([1.0, 0.0] + [0.0] * 766)  # Similar to query
        different_embedding = np.array([0.0, 1.0] + [0.0] * 766)  # Different from query

        memory_data = {
            b"stm_1": {
                b"content": b"This is about AI and embeddings",
                b"created_at": b"2025-07-07T12:00:00+00:00",
                b"id": b"stm_1",
                b"semantic_vector": json.dumps(similar_embedding.tolist()).encode(),
            },
            b"stm_2": {
                b"content": b"This is about cooking pasta",
                b"created_at": b"2025-07-07T11:00:00+00:00",
                b"id": b"stm_2",
                b"semantic_vector": json.dumps(different_embedding.tolist()).encode(),
            },
            b"stm_3": {
                b"content": b"Another memory about machine learning",
                b"created_at": b"2025-07-07T10:00:00+00:00",
                b"id": b"stm_3",
                b"semantic_vector": json.dumps(similar_embedding.tolist()).encode(),
            },
        }

        def mock_hgetall(key):
            memory_id = key.split(":")[1]
            return memory_data.get(memory_id.encode(), {})

        mock_redis.hgetall.side_effect = mock_hgetall

        # Query embedding similar to similar_embedding
        query_embedding = np.array([0.9, 0.1] + [0.0] * 766)

        results = search_related_memories("test query about AI", query_embedding)

        # Should return the two similar memories, sorted by similarity
        assert len(results) <= 5  # Limited to top 5
        assert len(results) >= 1  # Should find at least one similar memory

        # Verify all results have the expected structure
        for result in results:
            assert "content" in result
            assert "similarity_score" in result
            assert "created_at" in result
            assert "id" in result
            assert "source" in result
            assert result["source"] == "redis_search"
            assert result["similarity_score"] > 0.3  # Above threshold

    def test_exclude_memory_id(self, mock_redis):
        """Test that exclude_id parameter works correctly."""
        memory_ids = [b"stm_1", b"stm_2"]
        mock_redis.zrevrange.return_value = memory_ids

        # Mock memory data
        similar_embedding = np.array([1.0, 0.0] + [0.0] * 766)

        memory_data = {
            b"stm_1": {
                b"content": b"Memory 1",
                b"created_at": b"2025-07-07T12:00:00+00:00",
                b"id": b"stm_1",
                b"semantic_vector": json.dumps(similar_embedding.tolist()).encode(),
            },
            b"stm_2": {
                b"content": b"Memory 2",
                b"created_at": b"2025-07-07T11:00:00+00:00",
                b"id": b"stm_2",
                b"semantic_vector": json.dumps(similar_embedding.tolist()).encode(),
            },
        }

        def mock_hgetall(key):
            memory_id = key.split(":")[1]
            return memory_data.get(memory_id.encode(), {})

        mock_redis.hgetall.side_effect = mock_hgetall

        query_embedding = similar_embedding

        # Search excluding stm_1
        results = search_related_memories("test", query_embedding, exclude_id="stm_1")

        # Should only return stm_2
        assert len(results) == 1
        assert results[0]["id"] == "stm_2"

    def test_cosine_similarity_calculation(self, mock_redis):
        """Test that cosine similarity is calculated correctly."""
        memory_ids = [b"stm_1"]
        mock_redis.zrevrange.return_value = memory_ids

        # Create specific vectors for testing cosine similarity
        stored_vector = np.array([1.0, 0.0, 0.0])  # Unit vector along x-axis
        query_vector = np.array(
            [0.5, 0.866, 0.0]
        )  # 60-degree angle, should give cos(60°) = 0.5

        # Pad to expected dimensions
        stored_vector = np.pad(stored_vector, (0, 765))
        query_vector = np.pad(query_vector, (0, 765))

        memory_data = {
            b"stm_1": {
                b"content": b"Test memory",
                b"created_at": b"2025-07-07T12:00:00+00:00",
                b"id": b"stm_1",
                b"semantic_vector": json.dumps(stored_vector.tolist()).encode(),
            }
        }

        mock_redis.hgetall.return_value = memory_data[b"stm_1"]

        results = search_related_memories("test", query_vector)

        assert len(results) == 1
        # Should be approximately 0.5 (cos of 60 degrees)
        assert abs(results[0]["similarity_score"] - 0.5) < 0.01

    def test_error_handling(self, mock_redis):
        """Test error handling in search function."""
        # Make Redis operations raise an exception
        mock_redis.zrevrange.side_effect = Exception("Redis connection failed")

        query_embedding = np.random.rand(768).astype(np.float32)
        results = search_related_memories("test query", query_embedding)

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

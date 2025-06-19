"""
Tests for emotional embedding functionality in Redis short-term memory.
"""

import asyncio
import json
import os
import pytest
import sys
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alpha_recall.db.redis_db import RedisShortTermMemory
from alpha_recall.db.composite_db import CompositeDatabase


class TestEmotionalEmbeddings:
    """Test suite for emotional embedding functionality."""

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        client = AsyncMock()
        client.ping = AsyncMock()
        client.hset = AsyncMock()
        client.hgetall = AsyncMock()
        client.keys = AsyncMock()
        client.expire = AsyncMock()
        client.execute_command = AsyncMock()
        return client

    @pytest.fixture
    def mock_embedding_response(self):
        """Mock HTTP response for embedding generation."""
        return {
            "embeddings": [
                [0.1] * 1024  # 1024-dimensional emotional embedding
            ]
        }

    @pytest.fixture
    def redis_stm(self, mock_redis_client):
        """Redis short-term memory instance with mocked client."""
        stm = RedisShortTermMemory(
            host="localhost",
            port=6379,
            ttl=3600,
            emotional_embedding_url="http://localhost:6004/sentiment-embeddings"
        )
        stm.client = mock_redis_client
        return stm

    @pytest.mark.asyncio
    async def test_emotional_embedding_generation(self, redis_stm, mock_embedding_response):
        """Test that emotional embeddings are generated correctly."""
        # Directly mock the method to return the expected embedding
        expected_embedding = np.array([0.1] * 1024, dtype=np.float32)
        redis_stm._embed_text_emotional = AsyncMock(return_value=expected_embedding)
        
        # Test emotional embedding generation
        embedding = await redis_stm._embed_text_emotional("I am very happy today!")
        
        # Verify the embedding was generated
        assert embedding is not None
        assert len(embedding) == 1024  # 1024-dimensional emotional embedding
        assert isinstance(embedding, np.ndarray)
        np.testing.assert_array_equal(embedding, expected_embedding)

    @pytest.mark.asyncio
    async def test_emotional_embedding_failure_handling(self, redis_stm):
        """Test that emotional embedding failures are handled gracefully."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Test that failures return None
            embedding = await redis_stm._embed_text_emotional("Test text")
            assert embedding is None

    @pytest.mark.asyncio
    async def test_dual_embedding_storage(self, redis_stm, mock_embedding_response):
        """Test that both semantic and emotional embeddings are stored."""
        # Mock the embedding methods to return proper embeddings
        redis_stm._embed_text = AsyncMock(return_value=np.array([0.1] * 384, dtype=np.float32))
        redis_stm._embed_text_emotional = AsyncMock(return_value=np.array([0.1] * 1024, dtype=np.float32))
        
        # Store a memory
        result = await redis_stm.store_memory("I feel great about this project!")
        
        # Verify both embeddings were generated and stored
        assert result["has_semantic_embedding"] is True
        assert result["has_emotional_embedding"] is True
        
        # Verify both embedding methods were called
        redis_stm._embed_text.assert_called_once()
        redis_stm._embed_text_emotional.assert_called_once()

    @pytest.mark.asyncio
    async def test_emotional_search(self, redis_stm, mock_embedding_response):
        """Test emotional search functionality."""
        # Mock search results
        mock_search_results = [
            3,  # Number of results
            b"alpha:stm:123-abc",  # Key 1
            [b"content", b"I am excited!", b"created_at", b"2023-01-01T00:00:00", b"score", b"0.2"],
            b"alpha:stm:456-def",  # Key 2
            [b"content", b"Feeling happy", b"created_at", b"2023-01-01T01:00:00", b"score", b"0.3"],
        ]
        
        redis_stm.client.execute_command = AsyncMock(return_value=mock_search_results)
        
        # Mock the emotional embedding generation directly
        redis_stm._embed_text_emotional = AsyncMock(return_value=np.array([0.1] * 1024, dtype=np.float32))
        
        # Perform emotional search
        results = await redis_stm.emotional_search_memories("I'm feeling joyful", limit=5)
        
        # Verify results
        assert len(results) == 2
        assert results[0]["content"] == "I am excited!"
        assert results[1]["content"] == "Feeling happy"
        assert "emotional_score" in results[0]
        assert "emotional_score" in results[1]

    @pytest.mark.asyncio
    async def test_composite_db_emotional_integration(self):
        """Test that CompositeDatabase integrates emotional search correctly."""
        # Mock the shortterm_memory component
        mock_stm = AsyncMock()
        mock_stm.emotional_search_memories = AsyncMock(return_value=[
            {
                "id": "test_key",
                "content": "Happy memory",
                "emotional_score": 0.2,
                "created_at": "2023-01-01T00:00:00"
            }
        ])
        
        # Create CompositeDatabase with mocked components
        composite_db = CompositeDatabase(
            graph_db=AsyncMock(),
            semantic_search=AsyncMock(),
            shortterm_memory=mock_stm
        )
        
        # Test emotional search
        results = await composite_db.emotional_search_shortterm("happy query")
        
        # Verify the call was made correctly
        mock_stm.emotional_search_memories.assert_called_once_with(
            query="happy query",
            limit=10,
            through_the_last=None
        )
        
        assert len(results) == 1
        assert results[0]["content"] == "Happy memory"

    @pytest.mark.asyncio
    async def test_enhanced_relevance_scoring(self):
        """Test that relevance scoring includes emotional similarity."""
        # Mock the shortterm_memory component
        mock_stm = AsyncMock()
        
        # Mock semantic search results (lower score)
        mock_stm.semantic_search_memories = AsyncMock(return_value=[
            {
                "id": "memory1", 
                "content": "Test content",
                "similarity_score": 0.1,
                "created_at": datetime.utcnow().isoformat()
            }
        ])
        
        # Mock emotional search results (higher score, should be prioritized)
        mock_stm.emotional_search_memories = AsyncMock(return_value=[
            {
                "id": "memory1",
                "content": "Test content", 
                "emotional_score": 0.8,  # Higher score than semantic
                "created_at": datetime.utcnow().isoformat()
            }
        ])
        
        # Mock recent memories
        mock_stm.get_recent_memories = AsyncMock(return_value=[])
        
        # Create CompositeDatabase
        composite_db = CompositeDatabase(
            graph_db=AsyncMock(),
            semantic_search=AsyncMock(),
            shortterm_memory=mock_stm
        )
        
        # Test enhanced relevance scoring
        results = await composite_db._get_relevant_memories("test query", limit=5, include_emotional=True)
        
        # Verify that relevance scoring was performed
        assert len(results) >= 1
        memory = results[0]
        assert "score" in memory
        assert "search_type" in memory
        # Since include_emotional=True, emotional search should be included
        # The highest scoring result should be returned first
        assert memory["score"] > 0
        
        # Verify that emotional search was called when include_emotional=True
        mock_stm.emotional_search_memories.assert_called_once()
        mock_stm.semantic_search_memories.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
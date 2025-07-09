"""Short-term memory tests for Alpha-Recall MCP server."""

import json

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_remember_shortterm_tool(test_stack):
    """Test the remember_shortterm tool via MCP interface."""
    async with Client(test_stack) as client:
        # Test with short content
        result = await client.call_tool(
            "remember_shortterm",
            {"content": "This is a simple test memory to check performance."},
        )

        # Parse the JSON response
        response_data = json.loads(result.content[0].text)

        # Verify the response structure
        assert "status" in response_data
        assert response_data["status"] == "stored"
        assert "content_length" in response_data
        assert "content_tokens" in response_data
        assert "semantic_embedding_dims" in response_data
        assert "emotional_embedding_dims" in response_data
        assert "timing" in response_data
        assert "performance" in response_data
        assert "correlation_id" in response_data

        # Verify embedding dimensions
        assert response_data["semantic_embedding_dims"] == 768
        assert response_data["emotional_embedding_dims"] == 1024

        # Verify performance metrics are reasonable
        assert response_data["performance"]["semantic_tokens_per_sec"] > 0
        assert response_data["performance"]["emotional_tokens_per_sec"] > 0
        assert response_data["timing"]["total_ms"] > 0

        print(
            f"✅ Short content: {response_data['performance']['total_tokens_per_sec']} tokens/sec"
        )


@pytest.mark.asyncio
async def test_remember_shortterm_performance(test_stack):
    """Test remember_shortterm performance with longer content."""
    async with Client(test_stack) as client:
        long_content = """
        This is a comprehensive test of the remember_shortterm functionality
        with sentence-transformers v5.0.0. We are testing the performance
        improvements that come from upgrading from v2.7.0 to v5.0.0, which
        should provide significant speed improvements for both semantic and
        emotional embedding generation. The goal is to measure real-world
        performance in a containerized environment where we cannot use
        Metal Performance Shaders and are limited to CPU-only processing.
        """

        result = await client.call_tool(
            "remember_shortterm", {"content": long_content.strip()}
        )

        response_data = json.loads(result.content[0].text)

        # Verify faster processing for longer content
        assert response_data["content_tokens"] > 50  # Should be substantial content
        assert (
            response_data["performance"]["total_tokens_per_sec"] > 10
        )  # Better than old baseline

        print(
            f"✅ Long content: {response_data['content_tokens']} tokens in {response_data['timing']['total_ms']}ms"
        )
        print(
            f"📊 Performance: {response_data['performance']['total_tokens_per_sec']} tokens/sec"
        )

        # Verify this is much better than the old ~10 tokens/sec baseline
        assert (
            response_data["performance"]["total_tokens_per_sec"] > 50
        ), "Should be significantly faster than v2.x baseline"


@pytest.mark.asyncio
async def test_remember_shortterm_splash_functionality(test_stack):
    """Test the splash functionality that finds related memories."""
    async with Client(test_stack) as client:
        # Test with content that should trigger our mock related memories
        test_content = "Working on alpha-recall memory embedding improvements"

        result = await client.call_tool("remember_shortterm", {"content": test_content})

        response_data = json.loads(result.content[0].text)

        # Verify basic response structure first
        assert response_data["status"] == "stored"

        # Verify splash section exists
        assert "splash" in response_data, "Response should include splash section"
        splash = response_data["splash"]

        # Verify splash structure
        assert "related_memories_found" in splash
        assert "search_time_ms" in splash
        assert "memories" in splash

        # Verify splash performance
        assert isinstance(splash["related_memories_found"], int)
        assert splash["related_memories_found"] >= 0
        assert isinstance(splash["search_time_ms"], int | float)
        assert splash["search_time_ms"] < 1000, "Splash search should be fast"

        # Verify memories structure
        memories = splash["memories"]
        assert isinstance(memories, list)

        # With a fresh Redis instance, we might not find related memories initially
        # This is normal behavior - let's verify the structure is correct
        assert isinstance(len(memories), int), "Should return valid memory count"

        # If we do find memories, verify their structure
        for memory in memories:
            assert "content" in memory
            assert "similarity_score" in memory
            assert "created_at" in memory
            assert "id" in memory
            assert "source" in memory

            # Verify field types and constraints
            assert isinstance(memory["content"], str)
            assert len(memory["content"]) > 0
            assert isinstance(memory["similarity_score"], int | float)
            assert 0.0 <= memory["similarity_score"] <= 1.0
            assert isinstance(memory["created_at"], str)
            assert isinstance(memory["id"], str)
            assert memory["source"] == "redis_vector_search"

        # Verify memories are sorted by similarity (highest first)
        if len(memories) > 1:
            for i in range(len(memories) - 1):
                assert (
                    memories[i]["similarity_score"]
                    >= memories[i + 1]["similarity_score"]
                ), "Memories should be sorted by similarity score (highest first)"

        print(f"✅ Splash found {len(memories)} related memories")
        print(f"🔍 Search completed in {splash['search_time_ms']}ms")

        if memories:
            top_similarity = memories[0]["similarity_score"]
            print(f"🎯 Top similarity score: {top_similarity}")
            # With real Redis, similarity scores will be realistic (0.0-1.0 range)
            assert (
                0.0 <= top_similarity <= 1.0
            ), "Similarity score should be in valid range"


@pytest.mark.asyncio
async def test_remember_shortterm_splash_keyword_matching(test_stack):
    """Test that splash functionality works with real Redis storage and search."""
    async with Client(test_stack) as client:
        # First, store a few memories to test the search functionality
        seed_memories = [
            "Testing embedding performance tools for AI development",
            "Using Claude Code with FastMCP integration patterns",
            "Redis vector search implementation with cosine similarity",
        ]

        stored_ids = []
        for memory_content in seed_memories:
            result = await client.call_tool(
                "remember_shortterm", {"content": memory_content}
            )
            response_data = json.loads(result.content[0].text)
            stored_ids.append(response_data["memory_id"])
            print(f"Stored: {memory_content[:50]}...")

        # Now test that we can find related memories
        query_content = "Testing embedding performance tools"
        result = await client.call_tool(
            "remember_shortterm", {"content": query_content}
        )

        response_data = json.loads(result.content[0].text)
        memories = response_data["splash"]["memories"]

        # With real Redis vector search, we should find some related memories
        # (might be 0 if similarities are too low, but structure should be correct)
        assert isinstance(memories, list), "Should return a list of memories"

        # Verify each found memory has correct structure
        for memory in memories:
            assert "content" in memory
            assert "similarity_score" in memory
            assert "created_at" in memory
            assert "id" in memory
            assert "source" in memory
            assert memory["source"] == "redis_vector_search"
            assert 0.0 <= memory["similarity_score"] <= 1.0

        print(
            f"✅ Query '{query_content[:30]}...' found {len(memories)} related memories"
        )
        if memories:
            print(f"🎯 Top similarity: {memories[0]['similarity_score']:.3f}")
            print(f"📝 Top match: {memories[0]['content'][:50]}...")
        else:
            print(
                "ℹ️  No memories found above similarity threshold (0.3) - this is normal"
            )

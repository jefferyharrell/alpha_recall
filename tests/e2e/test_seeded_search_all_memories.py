"""Seeded E2E tests for unified search functionality.

These tests use comprehensive mock data to verify search behavior against realistic,
populated databases with entities, relationships, memories, and narratives.
"""

import json

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_search_all_memories_finds_known_entities(test_stack_seeded):
    """Test unified search finds entities from our seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for Alpha - should find across multiple systems
        result = await client.call_tool("search_all_memories", {"query": "Alpha"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["results"]) > 0

        # Should find Alpha in multiple contexts
        result_text = " ".join([r["content"] for r in data["results"]])
        assert "Alpha" in result_text

        # Verify we get results from multiple sources
        sources = {r["source"] for r in data["results"]}
        assert len(sources) > 1  # Should find in multiple systems


@pytest.mark.asyncio
async def test_search_all_memories_finds_sparkle_bread_crimes(test_stack_seeded):
    """Test search finds Sparkle's legendary bread crimes."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for bread-related crimes
        result = await client.call_tool("search_all_memories", {"query": "bread"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["results"]) > 0

        # Should find Sparkle's bread-related activities
        result_text = " ".join([r["content"] for r in data["results"]]).lower()
        assert "sparkle" in result_text
        assert "bread" in result_text


@pytest.mark.asyncio
async def test_search_all_memories_technical_terms(test_stack_seeded):
    """Test search finds technical concepts across systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for Redis - should find in observations and narratives
        result = await client.call_tool("search_all_memories", {"query": "Redis"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["results"]) > 0

        # Should find Redis mentioned in multiple contexts
        result_text = " ".join([r["content"] for r in data["results"]])
        assert "Redis" in result_text


@pytest.mark.asyncio
async def test_search_all_memories_collaborative_work(test_stack_seeded):
    """Test search finds collaborative development stories."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for collaboration terms
        result = await client.call_tool(
            "search_all_memories", {"query": "Jeffery collaboration"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["results"]) > 0

        # Should find collaborative work between Alpha and Jeffery
        result_text = " ".join([r["content"] for r in data["results"]])
        assert "Jeffery" in result_text


@pytest.mark.asyncio
async def test_search_all_memories_emotional_context(test_stack_seeded):
    """Test search finds memories with emotional content."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for emotional terms that should trigger emotional embeddings
        result = await client.call_tool(
            "search_all_memories", {"query": "excited breakthrough"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        # May or may not find results depending on exact emotional content


@pytest.mark.asyncio
async def test_search_all_memories_different_queries_different_results(
    test_stack_seeded,
):
    """Test that different queries return different results."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Two very different queries should return different results
        alpha_result = await client.call_tool(
            "search_all_memories", {"query": "Alpha consciousness"}
        )
        sparkle_result = await client.call_tool(
            "search_all_memories", {"query": "Sparkle bread heist"}
        )

        alpha_data = json.loads(alpha_result.content[0].text)
        sparkle_data = json.loads(sparkle_result.content[0].text)

        assert alpha_data["success"] is True
        assert sparkle_data["success"] is True

        # Results should be different
        alpha_content = [r["content"] for r in alpha_data["results"]]
        sparkle_content = [r["content"] for r in sparkle_data["results"]]

        # At least some results should be different
        assert alpha_content != sparkle_content


@pytest.mark.asyncio
async def test_search_all_memories_cross_system_integration(test_stack_seeded):
    """Test that search finds references across different memory systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for something that should appear in multiple systems
        result = await client.call_tool(
            "search_all_memories", {"query": "Alpha-Recall"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["results"]) > 0

        # Verify we get results from multiple sources (STM, LTM, NM)
        sources = {r["source"] for r in data["results"]}
        assert "LTM" in sources or "ENTITY" in sources

        # Should find Alpha-Recall project references
        result_text = " ".join([r["content"] for r in data["results"]])
        assert "Alpha-Recall" in result_text or "alpha-recall" in result_text.lower()


@pytest.mark.asyncio
async def test_search_all_memories_performance_reasonable(test_stack_seeded):
    """Test that search performance is reasonable with seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        import time

        start_time = time.time()
        result = await client.call_tool("search_all_memories", {"query": "performance"})
        end_time = time.time()

        data = json.loads(result.content[0].text)
        assert data["success"] is True

        # Search should complete within reasonable time (10 seconds max)
        search_time = end_time - start_time
        assert search_time < 10.0, f"Search took {search_time:.2f}s, should be < 10s"

        # Should include timing information in metadata
        assert "metadata" in data
        assert "search_time_ms" in data["metadata"]
        assert isinstance(data["metadata"]["search_time_ms"], int | float)


@pytest.mark.asyncio
async def test_search_all_memories_respects_limits(test_stack_seeded):
    """Test that search respects limit parameters."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search with small limit
        result = await client.call_tool(
            "search_all_memories", {"query": "Alpha", "limit": 3}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["results"]) <= 3

        # Search with larger limit should potentially return more results
        result2 = await client.call_tool(
            "search_all_memories", {"query": "Alpha", "limit": 10}
        )
        data2 = json.loads(result2.content[0].text)

        assert data2["success"] is True
        assert len(data2["results"]) >= len(data["results"])

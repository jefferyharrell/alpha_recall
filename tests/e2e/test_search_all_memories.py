"""E2E tests for search_all_memories tool."""

import asyncio
import json

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_search_all_memories_different_queries_return_different_results(
    test_stack,
):
    """Test that different queries return different results (bug reproduction test)."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Store some test memories with specific content
        await client.call_tool(
            "remember_shortterm", {"content": "I love pizza with pepperoni"}
        )
        await client.call_tool(
            "remember_shortterm", {"content": "Tagline is a project management tool"}
        )
        await client.call_tool(
            "remember_shortterm", {"content": "Sparkle loves tuna fish"}
        )

        # Wait a moment for indexing
        await asyncio.sleep(1)

        # Search for different queries
        pizza_result = await client.call_tool(
            "search_all_memories", {"query": "pizza pepperoni", "limit": 10}
        )
        tagline_result = await client.call_tool(
            "search_all_memories", {"query": "Tagline project", "limit": 10}
        )
        sparkle_result = await client.call_tool(
            "search_all_memories", {"query": "Sparkle tuna", "limit": 10}
        )

        # Parse results
        pizza_data = json.loads(pizza_result.content[0].text)
        tagline_data = json.loads(tagline_result.content[0].text)
        sparkle_data = json.loads(sparkle_result.content[0].text)

        # All searches should succeed
        assert pizza_data["success"] is True
        assert tagline_data["success"] is True
        assert sparkle_data["success"] is True

        # Extract result content for comparison
        pizza_contents = [r["content"] for r in pizza_data["results"]]
        tagline_contents = [r["content"] for r in tagline_data["results"]]
        sparkle_contents = [r["content"] for r in sparkle_data["results"]]

        # Different queries should return different results
        # This test will FAIL before the bug fix and PASS after
        assert (
            pizza_contents != tagline_contents
        ), "Pizza and Tagline searches should return different results"
        assert (
            pizza_contents != sparkle_contents
        ), "Pizza and Sparkle searches should return different results"
        assert (
            tagline_contents != sparkle_contents
        ), "Tagline and Sparkle searches should return different results"

        # Verify query-specific content appears in relevant results
        pizza_text = " ".join(pizza_contents).lower()
        tagline_text = " ".join(tagline_contents).lower()
        sparkle_text = " ".join(sparkle_contents).lower()

        # At least one result should contain the query terms
        assert (
            "pizza" in pizza_text or "pepperoni" in pizza_text
        ), "Pizza search should find pizza-related content"
        assert (
            "tagline" in tagline_text or "project" in tagline_text
        ), "Tagline search should find tagline-related content"
        assert (
            "sparkle" in sparkle_text or "tuna" in sparkle_text
        ), "Sparkle search should find sparkle-related content"


@pytest.mark.asyncio
async def test_search_all_memories_nonexistent_query_returns_no_matches(test_stack):
    """Test that searching for non-existent content returns appropriate results."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Search for something that definitely doesn't exist
        result = await client.call_tool(
            "search_all_memories", {"query": "xyzzyx_impossible_term_12345", "limit": 5}
        )

        data = json.loads(result.content[0].text)
        assert data["success"] is True

        # Should return very few or no results
        assert (
            len(data["results"]) <= 5
        ), "Should not return many results for non-existent terms"

        # If there are results, they should have low scores
        if data["results"]:
            for result in data["results"]:
                # Scores should be relatively low for non-matching content
                assert (
                    result["score"] < 0.8
                ), f"Score {result['score']} too high for non-matching query"


@pytest.mark.asyncio
async def test_search_all_memories_covers_all_sources(test_stack):
    """Test that search_all_memories searches across all memory systems."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Add memories to different systems
        await client.call_tool(
            "remember_shortterm", {"content": "STM test memory about dogs"}
        )
        await client.call_tool(
            "remember_longterm",
            {"entity": "Test Entity", "observation": "LTM test observation about cats"},
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Search for terms that should match different systems
        result = await client.call_tool(
            "search_all_memories", {"query": "test", "limit": 20}
        )

        data = json.loads(result.content[0].text)
        assert data["success"] is True

        # Verify metadata indicates all sources were searched
        expected_sources = [
            "STM_SEMANTIC",
            "STM_EMOTIONAL",
            "LTM",
            "NM_SEMANTIC",
            "NM_EMOTIONAL",
            "ENTITIES",
        ]
        assert data["metadata"]["sources_searched"] == expected_sources

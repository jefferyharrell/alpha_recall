"""Integration tests using mock data to validate STM functionality."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Add src to path so we can import alpha_recall
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from alpha_recall.config import AlphaRecallSettings


@pytest.fixture(scope="session")
def dev_server_url():
    """Get the dev server URL from configuration."""
    config = AlphaRecallSettings()
    return f"http://localhost:{config.alpha_recall_dev_port}/mcp/"


@pytest.fixture
def mock_stm_data():
    """Load mock STM data from JSON file."""
    data_file = Path(__file__).parent.parent.parent / "mock_data" / "stm_test_data.json"
    with open(data_file) as f:
        data = json.load(f)
    return data["memories"]


@pytest.mark.asyncio
async def test_load_and_search_mock_data(mock_stm_data, dev_server_url):
    """Load mock data and test semantic search functionality."""
    async with streamablehttp_client(dev_server_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Load a subset of memories to avoid overwhelming the test
            test_memories = mock_stm_data[:10]  # Just first 10 memories
            stored_ids = []

            # Store memories
            for memory_data in test_memories:
                result = await session.call_tool(
                    "remember_shortterm", {"content": memory_data["content"]}
                )

                result_data = json.loads(result.content[0].text)
                assert (
                    result_data.get("status") == "stored"
                ), f"Failed to store: {memory_data['content'][:50]}"
                stored_ids.append(result_data["memory_id"])

                # Check splash functionality on first few memories
                if len(stored_ids) <= 3:
                    splash = result_data.get("splash", {})
                    assert (
                        "related_memories_found" in splash
                    ), "Should include splash metadata"
                    assert "search_time_ms" in splash, "Should include search timing"

                # Small delay
                await asyncio.sleep(0.1)

            # Test semantic search
            search_result = await session.call_tool(
                "search_shortterm",
                {
                    "query": "redis performance vector search",
                    "limit": 5,
                    "search_type": "semantic",
                },
            )

            search_data = json.loads(search_result.content[0].text)
            assert "memories" in search_data, "Search should return memories"
            assert "search_metadata" in search_data, "Search should include metadata"

            memories = search_data["memories"]
            if memories:
                # Check similarity scores
                for memory in memories:
                    assert "similarity_score" in memory, "Should have similarity score"
                    assert (
                        0 <= memory["similarity_score"] <= 1
                    ), "Similarity score should be 0-1"

                # Check that we found relevant content
                redis_found = any("redis" in m["content"].lower() for m in memories)
                performance_found = any(
                    any(
                        term in m["content"].lower()
                        for term in ["performance", "fast", "speed"]
                    )
                    for m in memories
                )

                # Should find at least some relevant content
                assert (
                    redis_found or performance_found
                ), "Should find Redis or performance related content"


@pytest.mark.asyncio
async def test_emotional_search_functionality(mock_stm_data, dev_server_url):
    """Test emotional search with mock data."""
    async with streamablehttp_client(dev_server_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Store a few memories with different emotional tones
            emotional_memories = [
                "Successfully implemented the new feature! Everything is working perfectly and I'm really excited about the results.",
                "Frustrated with this debugging session. Nothing seems to work and I'm running out of ideas.",
                "The performance benchmarks look excellent. We're hitting all our targets.",
            ]

            for content in emotional_memories:
                await session.call_tool("remember_shortterm", {"content": content})
                await asyncio.sleep(0.1)

            # Test emotional search
            result = await session.call_tool(
                "search_shortterm",
                {
                    "query": "frustrated disappointed worried",
                    "limit": 5,
                    "search_type": "emotional",
                },
            )

            search_data = json.loads(result.content[0].text)
            assert "memories" in search_data, "Emotional search should return memories"
            assert (
                search_data["search_metadata"]["search_type"] == "emotional"
            ), "Should preserve search type"


@pytest.mark.asyncio
async def test_browse_functionality_with_mock_data(mock_stm_data, dev_server_url):
    """Test browse functionality with loaded mock data."""
    async with streamablehttp_client(dev_server_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Store a few test memories
            test_contents = [
                "Testing browse functionality with Redis optimization work",
                "Another memory about performance improvements and debugging",
                "Final test memory with different content about embeddings",
            ]

            for content in test_contents:
                await session.call_tool("remember_shortterm", {"content": content})
                await asyncio.sleep(0.1)

            # Test browsing
            result = await session.call_tool("browse_shortterm", {"limit": 10})

            browse_data = json.loads(result.content[0].text)
            assert "memories" in browse_data, "Browse should return memories"
            assert "pagination" in browse_data, "Browse should include pagination"

            memories = browse_data["memories"]
            if memories:
                # Check that memories have required fields
                for memory in memories:
                    assert "content" in memory, "Memory should have content"
                    assert "created_at" in memory, "Memory should have timestamp"
                    assert "age" in memory, "Memory should have human-readable age"

            # Test browse with search filter
            result = await session.call_tool(
                "browse_shortterm", {"limit": 10, "search": "redis"}
            )

            browse_data = json.loads(result.content[0].text)
            filtered_memories = browse_data["memories"]

            # All filtered memories should contain "redis"
            for memory in filtered_memories:
                assert (
                    "redis" in memory["content"].lower()
                ), f"Filtered memory should contain 'redis': {memory['content']}"


@pytest.mark.asyncio
async def test_splash_excludes_self(dev_server_url):
    """Test that splash functionality doesn't return the memory we just stored."""
    async with streamablehttp_client(dev_server_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Store a memory with unique content
            unique_content = f"Unique test memory for splash validation {asyncio.get_event_loop().time()}"

            result = await session.call_tool(
                "remember_shortterm", {"content": unique_content}
            )

            result_data = json.loads(result.content[0].text)
            stored_id = result_data["memory_id"]

            # Check splash results
            splash = result_data["splash"]
            splash_memories = splash["memories"]
            splash_ids = [m.get("id") for m in splash_memories]

            # The memory we just stored should not appear in splash results
            assert (
                stored_id not in splash_ids
            ), "Splash should not include the memory we just stored"

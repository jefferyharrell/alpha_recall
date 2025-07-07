"""Integration tests for short-term memory functionality.

These tests connect to the actual running dev container to verify
the remember_shortterm tool works correctly with real Redis storage.
Tests the full MCP protocol integration.
"""

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


@pytest.mark.asyncio
async def test_remember_shortterm_tool_available(dev_server_url):
    """Test that the remember_shortterm tool is available via MCP."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # List available tools
                result = await session.list_tools()
                tool_names = [tool.name for tool in result.tools]

                # Check that remember_shortterm tool is available
                assert "remember_shortterm" in tool_names

                # Get the tool definition
                tool_def = next(
                    tool for tool in result.tools if tool.name == "remember_shortterm"
                )

                # Verify the tool has the expected parameter
                assert tool_def.inputSchema is not None
                schema = tool_def.inputSchema
                assert "content" in schema.get("properties", {})

    except Exception as e:
        pytest.fail(f"Failed to connect to dev server at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_remember_shortterm_basic_storage(dev_server_url):
    """Test basic memory storage functionality via MCP."""
    test_content = "This is a test memory for integration testing with Redis storage"

    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call the remember_shortterm tool
                result = await session.call_tool(
                    "remember_shortterm", {"content": test_content}
                )

                # Parse the JSON response
                assert result.content is not None
                text_content = result.content[0].text
                response_data = json.loads(text_content)

                # Verify basic response structure
                assert "status" in response_data
                assert response_data["status"] in ["stored", "processed_only"]

                # Verify memory metadata
                assert "memory_id" in response_data
                assert "created_at" in response_data
                assert "content_length" in response_data
                assert response_data["content_length"] == len(test_content)

                # Verify embedding dimensions
                assert "semantic_embedding_dims" in response_data
                assert "emotional_embedding_dims" in response_data
                assert response_data["semantic_embedding_dims"] == 768
                assert response_data["emotional_embedding_dims"] == 1024

                # Verify timing data
                assert "timing" in response_data
                timing = response_data["timing"]
                assert "total_ms" in timing
                assert "semantic_ms" in timing
                assert "emotional_ms" in timing
                assert "storage_ms" in timing
                assert all(isinstance(timing[key], int | float) for key in timing)

                # Verify performance metrics
                assert "performance" in response_data
                performance = response_data["performance"]
                assert "semantic_tokens_per_sec" in performance
                assert "emotional_tokens_per_sec" in performance
                assert "total_tokens_per_sec" in performance

                # Verify storage info
                assert "storage" in response_data
                storage = response_data["storage"]
                assert "success" in storage
                assert "backend" in storage
                assert storage["backend"] == "redis"

                # Verify splash functionality
                assert "splash" in response_data
                splash = response_data["splash"]
                assert "related_memories_found" in splash
                assert "search_time_ms" in splash
                assert "memories" in splash
                assert isinstance(splash["memories"], list)

    except Exception as e:
        pytest.fail(
            f"Failed to test remember_shortterm via MCP at {dev_server_url}: {e}"
        )


@pytest.mark.asyncio
async def test_remember_shortterm_splash_functionality(dev_server_url):
    """Test that splash functionality returns related memories."""
    # Store multiple related memories
    test_memories = [
        "Working on Redis vector search implementation",
        "Testing memory storage with embeddings",
        "Implementing cosine similarity search",
        "Debugging Redis connection issues",
    ]

    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Store multiple memories
                stored_memory_ids = []
                for memory_content in test_memories:
                    result = await session.call_tool(
                        "remember_shortterm", {"content": memory_content}
                    )

                    response_data = json.loads(result.content[0].text)
                    assert response_data["status"] == "stored"
                    stored_memory_ids.append(response_data["memory_id"])

                # Store one more memory that should find related ones
                query_memory = (
                    "Developing vector search algorithms for memory retrieval"
                )
                result = await session.call_tool(
                    "remember_shortterm", {"content": query_memory}
                )

                response_data = json.loads(result.content[0].text)

                # Check splash results
                splash = response_data["splash"]
                assert splash["related_memories_found"] >= 0

                # If we found related memories, verify their structure
                if splash["related_memories_found"] > 0:
                    memories = splash["memories"]
                    assert len(memories) <= 5  # Should be limited to top 5

                    for memory in memories:
                        assert "content" in memory
                        assert "similarity_score" in memory
                        assert "created_at" in memory
                        assert "id" in memory
                        assert "source" in memory
                        assert memory["source"] == "redis_vector_search"
                        assert 0.0 <= memory["similarity_score"] <= 1.0

                        # Verify we're not getting the memory we just stored
                        assert memory["id"] != response_data["memory_id"]

                    # Verify memories are sorted by similarity (highest first)
                    similarities = [m["similarity_score"] for m in memories]
                    assert similarities == sorted(similarities, reverse=True)

    except Exception as e:
        pytest.fail(
            f"Failed to test splash functionality via MCP at {dev_server_url}: {e}"
        )


@pytest.mark.asyncio
async def test_remember_shortterm_performance_requirements(dev_server_url):
    """Test that memory storage meets basic performance requirements."""
    test_content = "Performance test memory with sufficient content to measure embedding generation speed and storage latency in Redis"

    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call the remember_shortterm tool
                result = await session.call_tool(
                    "remember_shortterm", {"content": test_content}
                )

                response_data = json.loads(result.content[0].text)

                # Verify reasonable performance (these are loose requirements)
                timing = response_data["timing"]

                # Total time should be reasonable (less than 10 seconds)
                assert (
                    timing["total_ms"] < 10000
                ), f"Total time too slow: {timing['total_ms']}ms"

                # Storage should be fast (less than 1 second)
                assert (
                    timing["storage_ms"] < 1000
                ), f"Storage too slow: {timing['storage_ms']}ms"

                # Splash search should be fast (less than 1 second)
                splash = response_data["splash"]
                assert (
                    splash["search_time_ms"] < 1000
                ), f"Search too slow: {splash['search_time_ms']}ms"

                # Performance metrics should be reasonable
                performance = response_data["performance"]
                assert performance["total_tokens_per_sec"] > 0
                assert performance["semantic_tokens_per_sec"] > 0
                assert performance["emotional_tokens_per_sec"] > 0

    except Exception as e:
        pytest.fail(f"Failed to test performance via MCP at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_remember_shortterm_error_handling(dev_server_url):
    """Test error handling with invalid inputs via MCP."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Test with empty content
                try:
                    result = await session.call_tool(
                        "remember_shortterm", {"content": ""}
                    )

                    # Should handle empty content gracefully
                    response_data = json.loads(result.content[0].text)
                    assert "status" in response_data

                except Exception:
                    # It's acceptable for empty content to raise an error
                    pass

                # Test with missing content parameter
                try:
                    await session.call_tool("remember_shortterm", {})
                except Exception:
                    # This should raise an error due to missing required parameter
                    pass

    except Exception as e:
        pytest.fail(f"Failed to test error handling via MCP at {dev_server_url}: {e}")

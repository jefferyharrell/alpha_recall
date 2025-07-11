"""Greenfield E2E tests for basic health and bootstrap scenarios.

These tests verify that Alpha-Recall works correctly with fresh, empty databases.
They represent the "first run" experience and basic connectivity.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_00_warm_up_embedding_models(test_stack):
    """Warm up embedding models by triggering first search and verify load time.

    This test intentionally runs first (test_00_) to trigger model loading.
    It searches for a control word that won't exist in the fresh database,
    but will force both semantic and emotional embedding models to load.

    Subsequent tests can then assert faster response times since models
    will already be cached in memory.
    """
    server_url = test_stack
    async with Client(server_url) as client:
        # Search for control word that definitely won't exist in fresh database
        # This will trigger embedding model loading for both semantic and emotional
        result = await time_mcp_call(
            client, "search_all_memories", {"query": "Sparkle"}
        )

        data = json.loads(result.content[0].text)

        # Should complete successfully even with model loading
        assert data["success"] is True
        assert isinstance(data["results"], list)

        # Should return no results on fresh database
        assert data["metadata"]["total_found"] == 0
        assert len(data["results"]) == 0

        # The important assertion: model loading should complete within reasonable time
        # This includes both semantic and emotional model loading
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 10000
        ), f"Model loading took {latest_duration:.1f}ms, should be <10000ms"

        print(
            f"🔥 Model warm-up completed in {latest_duration:.1f}ms - models now cached!"
        )


@pytest.mark.asyncio
async def test_mcp_server_startup_and_health(test_stack):
    """Test that the MCP server starts up and responds to health checks."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Test basic MCP connectivity
        result = await client.call_tool("health_check")
        data = json.loads(result.content[0].text)

        assert data["status"] == "ok"
        assert "version" in data
        assert "checks" in data
        assert "timestamp" in data

        # Verify critical infrastructure services are healthy
        checks = data["checks"]
        assert "redis" in checks
        assert "memgraph" in checks
        assert checks["redis"] == "ok"
        assert checks["memgraph"] == "ok"


@pytest.mark.asyncio
async def test_gentle_refresh_with_empty_database(test_stack):
    """Test gentle_refresh works with completely empty databases."""
    server_url = test_stack
    async with Client(server_url) as client:
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "time" in data
        assert "core_identity" in data
        assert "personality" in data
        assert "shortterm_memories" in data
        assert "recent_observations" in data

        # With empty databases, these should be empty
        assert data["personality"] == {}
        assert data["shortterm_memories"] == []
        assert data["recent_observations"] == []


@pytest.mark.asyncio
@performance_test
async def test_first_memory_storage_works(test_stack):
    """Test that the very first memory can be stored successfully.

    Since test_00_warm_up_embedding_models ran first, models should already
    be loaded and this memory storage should be fast.
    """
    server_url = test_stack
    async with Client(server_url) as client:
        # Store the first ever memory in this fresh database
        result = await time_mcp_call(
            client,
            "remember_shortterm",
            {"content": "This is the first memory in a fresh Alpha-Recall system"},
        )

        data = json.loads(result.content[0].text)
        assert data["status"] == "stored"
        assert "memory_id" in data
        assert len(data["memory_id"]) > 0

        # Assert fast performance since models should already be loaded
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_remember_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for remember_shortterm"
        assert (
            latest_duration < 1500
        ), f"Memory storage took {latest_duration:.1f}ms, should be <1500ms with warm models"

        print(f"💾 Memory stored in {latest_duration:.1f}ms - models already warm!")


@pytest.mark.asyncio
async def test_first_entity_creation_works(test_stack):
    """Test that the first entity can be created successfully."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Create the first ever entity in this fresh database
        result = await client.call_tool(
            "remember_longterm",
            {
                "entity": "Alpha Test Entity",
                "type": "Test",
                "observation": "This is the first entity in a fresh Alpha-Recall system",
            },
        )

        data = json.loads(result.content[0].text)
        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Alpha Test Entity"


@pytest.mark.asyncio
@performance_test
async def test_search_returns_gracefully_for_nonsensical_queries(test_stack):
    """Test that unified search handles nonsensical queries gracefully.

    Since test_00_warm_up_embedding_models ran first, models should already
    be loaded and this test should run much faster.
    """
    server_url = test_stack
    async with Client(server_url) as client:
        # Test search tool handles completely nonsensical query gracefully
        unified_result = await time_mcp_call(
            client,
            "search_all_memories",
            {"query": "zxyqwerty_impossible_nonsense_12345_abcdef"},
        )
        unified_data = json.loads(unified_result.content[0].text)
        assert unified_data["success"] is True
        assert isinstance(unified_data["results"], list)
        # Should return very few results for nonsensical query
        assert len(unified_data["results"]) <= 3

        # Assert fast performance since models should already be loaded
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 1000
        ), f"Search took {latest_duration:.1f}ms, should be <1000ms with warm models"

        # Test the theory: second search should also be fast since models are loaded
        unified_result2 = await time_mcp_call(
            client,
            "search_all_memories",
            {"query": "another_nonsense_query_654321"},
        )
        unified_data2 = json.loads(unified_result2.content[0].text)
        assert unified_data2["success"] is True

        # Second search should also be fast
        latest_duration2 = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration2 = metric["duration_ms"]
                break

        assert (
            latest_duration2 is not None
        ), "Should have recorded timing for second search"
        assert (
            latest_duration2 < 500
        ), f"Second search took {latest_duration2:.1f}ms, should be <500ms"

        print(
            f"🚀 Warm searches: {latest_duration:.1f}ms and {latest_duration2:.1f}ms - models cached!"
        )

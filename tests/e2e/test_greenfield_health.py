"""Greenfield E2E tests for basic health and bootstrap scenarios.

These tests verify that Alpha-Recall works correctly with fresh, empty databases.
They represent the "first run" experience and basic connectivity.
"""

import json

import pytest
from fastmcp import Client


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
        assert "shortterm_memories" in data
        assert "recent_observations" in data

        # With empty databases, these should be empty
        assert data["shortterm_memories"] == []
        assert data["recent_observations"] == []


@pytest.mark.asyncio
async def test_first_memory_storage_works(test_stack):
    """Test that the very first memory can be stored successfully."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Store the first ever memory in this fresh database
        result = await client.call_tool(
            "remember_shortterm",
            {"content": "This is the first memory in a fresh Alpha-Recall system"},
        )

        data = json.loads(result.content[0].text)
        assert data["status"] == "stored"
        assert "memory_id" in data
        assert len(data["memory_id"]) > 0


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
async def test_search_returns_gracefully_for_nonsensical_queries(test_stack):
    """Test that unified search handles nonsensical queries gracefully."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Test search tool handles completely nonsensical query gracefully
        unified_result = await client.call_tool(
            "search_all_memories",
            {"query": "zxyqwerty_impossible_nonsense_12345_abcdef"},
        )
        unified_data = json.loads(unified_result.content[0].text)
        assert unified_data["success"] is True
        assert isinstance(unified_data["results"], list)
        # Should return very few results for nonsensical query
        assert len(unified_data["results"]) <= 3

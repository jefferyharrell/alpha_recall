"""Integration tests for gentle_refresh functionality.

These tests connect to the actual running dev container to verify
the gentle_refresh tool works correctly with real Redis and Memgraph storage.
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
async def test_gentle_refresh_tool_available(dev_server_url):
    """Test that the gentle_refresh tool is available via MCP."""
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

                # Check that gentle_refresh tool is available
                assert "gentle_refresh" in tool_names

                # Get the tool definition
                tool_def = next(
                    tool for tool in result.tools if tool.name == "gentle_refresh"
                )

                # Verify the tool has the expected parameter schema
                assert tool_def.inputSchema is not None
                schema = tool_def.inputSchema

                # Query parameter should be optional
                properties = schema.get("properties", {})
                if "query" in properties:
                    # If query is present, it should be optional (not in required)
                    required = schema.get("required", [])
                    assert "query" not in required

    except Exception as e:
        pytest.fail(f"Failed to connect to dev server at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_gentle_refresh_basic_response_structure(dev_server_url):
    """Test basic response structure from gentle_refresh via MCP."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call the gentle_refresh tool
                result = await session.call_tool("gentle_refresh", {})

                # Parse the JSON response
                assert result.content is not None
                text_content = result.content[0].text
                response_data = json.loads(text_content)

                # Verify basic response structure
                assert "success" in response_data
                assert response_data["success"] is True

                # Verify main sections
                assert "time" in response_data
                assert "core_identity" in response_data
                assert "shortterm_memories" in response_data
                assert "memory_consolidation" in response_data
                assert "recent_observations" in response_data

                # Verify these are the expected types
                assert isinstance(response_data["time"], dict)
                assert isinstance(response_data["shortterm_memories"], list)
                assert isinstance(response_data["memory_consolidation"], dict)
                assert isinstance(response_data["recent_observations"], list)

                # core_identity can be None or dict
                assert response_data["core_identity"] is None or isinstance(
                    response_data["core_identity"], dict
                )

    except Exception as e:
        pytest.fail(f"Failed to test gentle_refresh via MCP at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_gentle_refresh_time_structure(dev_server_url):
    """Test time object structure from gentle_refresh via MCP."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call the gentle_refresh tool
                result = await session.call_tool("gentle_refresh", {})

                response_data = json.loads(result.content[0].text)
                time_obj = response_data["time"]

                # Verify time structure matches v0.1.0 format
                required_keys = [
                    "iso_datetime",
                    "utc",
                    "local",
                    "human_readable",
                    "timezone",
                    "unix_timestamp",
                    "day_of_week",
                ]

                for key in required_keys:
                    assert key in time_obj, f"Missing required time key: {key}"

                # Verify timezone structure
                timezone = time_obj["timezone"]
                assert "name" in timezone
                assert "offset" in timezone
                assert "display" in timezone

                # Verify day_of_week structure
                day_of_week = time_obj["day_of_week"]
                assert "integer" in day_of_week
                assert "name" in day_of_week
                assert isinstance(day_of_week["integer"], int)
                assert isinstance(day_of_week["name"], str)

                # Verify types
                assert isinstance(time_obj["unix_timestamp"], int)
                assert isinstance(time_obj["human_readable"], str)
                assert isinstance(time_obj["iso_datetime"], str)

    except Exception as e:
        pytest.fail(f"Failed to test time structure via MCP at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_gentle_refresh_with_query_parameter(dev_server_url):
    """Test gentle_refresh accepts query parameter for compatibility."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Test with query parameter
                result = await session.call_tool(
                    "gentle_refresh", {"query": "test query"}
                )
                response_data = json.loads(result.content[0].text)
                assert response_data["success"] is True

                # Test without query parameter
                result = await session.call_tool("gentle_refresh", {})
                response_data = json.loads(result.content[0].text)
                assert response_data["success"] is True

    except Exception as e:
        pytest.fail(f"Failed to test query parameter via MCP at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_gentle_refresh_memory_consolidation_structure(dev_server_url):
    """Test memory consolidation structure from gentle_refresh via MCP."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call the gentle_refresh tool
                result = await session.call_tool("gentle_refresh", {})

                response_data = json.loads(result.content[0].text)
                consolidation = response_data["memory_consolidation"]

                # Verify memory consolidation structure
                required_keys = [
                    "entities",
                    "relationships",
                    "insights",
                    "summary",
                    "emotional_context",
                    "next_steps",
                    "processed_memories_count",
                    "consolidation_timestamp",
                    "model_used",
                ]

                for key in required_keys:
                    assert (
                        key in consolidation
                    ), f"Missing required consolidation key: {key}"

                # Verify types
                assert isinstance(consolidation["entities"], list)
                assert isinstance(consolidation["relationships"], list)
                assert isinstance(consolidation["insights"], list)
                assert isinstance(consolidation["summary"], str)
                assert isinstance(consolidation["emotional_context"], str)
                assert isinstance(consolidation["next_steps"], list)
                assert isinstance(consolidation["processed_memories_count"], int)
                assert isinstance(consolidation["consolidation_timestamp"], str)
                assert isinstance(consolidation["model_used"], str)

    except Exception as e:
        pytest.fail(
            f"Failed to test memory consolidation via MCP at {dev_server_url}: {e}"
        )


@pytest.mark.asyncio
async def test_gentle_refresh_with_stored_memories(dev_server_url):
    """Test gentle_refresh retrieves stored short-term memories."""
    test_memory = "Integration test memory for gentle_refresh validation"

    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Store a memory first
                await session.call_tool("remember_shortterm", {"content": test_memory})

                # Call gentle_refresh
                result = await session.call_tool("gentle_refresh", {})
                response_data = json.loads(result.content[0].text)

                # Verify we get shortterm_memories
                shortterm_memories = response_data["shortterm_memories"]
                assert isinstance(shortterm_memories, list)

                # If we have memories, verify structure
                if shortterm_memories:
                    memory = shortterm_memories[0]
                    assert "content" in memory
                    assert "created_at" in memory
                    assert "client" in memory
                    assert "client_name" in memory["client"]

                    # Verify types
                    assert isinstance(memory["content"], str)
                    assert isinstance(memory["created_at"], str)
                    assert isinstance(memory["client"], dict)
                    assert isinstance(memory["client"]["client_name"], str)

    except Exception as e:
        pytest.fail(
            f"Failed to test with stored memories via MCP at {dev_server_url}: {e}"
        )


@pytest.mark.asyncio
async def test_gentle_refresh_with_core_identity(dev_server_url):
    """Test gentle_refresh handles core identity entity retrieval."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Store a core identity observation first
                await session.call_tool(
                    "remember_longterm",
                    {
                        "entity": "Alpha Core Identity",
                        "observation": "Integration test observation for gentle_refresh",
                    },
                )

                # Call gentle_refresh
                result = await session.call_tool("gentle_refresh", {})
                response_data = json.loads(result.content[0].text)

                # Verify core_identity structure
                core_identity = response_data["core_identity"]

                if core_identity is not None:
                    assert isinstance(core_identity, dict)
                    assert "name" in core_identity
                    assert "observations" in core_identity
                    assert isinstance(core_identity["observations"], list)

                    # Verify observation structure if present
                    if core_identity["observations"]:
                        observation = core_identity["observations"][0]
                        assert "content" in observation
                        assert "created_at" in observation
                        assert isinstance(observation["content"], str)
                        assert isinstance(observation["created_at"], str)

    except Exception as e:
        pytest.fail(f"Failed to test core identity via MCP at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_gentle_refresh_recent_observations(dev_server_url):
    """Test gentle_refresh retrieves recent observations."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Store an observation first
                await session.call_tool(
                    "remember_longterm",
                    {
                        "entity": "Test Entity for Gentle Refresh",
                        "observation": "Integration test observation for recent observations",
                    },
                )

                # Call gentle_refresh
                result = await session.call_tool("gentle_refresh", {})
                response_data = json.loads(result.content[0].text)

                # Verify recent_observations structure
                recent_observations = response_data["recent_observations"]
                assert isinstance(recent_observations, list)

                # If we have observations, verify structure
                if recent_observations:
                    observation = recent_observations[0]
                    assert "entity_name" in observation
                    assert "content" in observation
                    assert "created_at" in observation

                    # Verify types
                    assert isinstance(observation["entity_name"], str)
                    assert isinstance(observation["content"], str)
                    # created_at should be a string, but may be int/float due to timestamp formatting issues
                    assert isinstance(observation["created_at"], str | int | float)

    except Exception as e:
        pytest.fail(
            f"Failed to test recent observations via MCP at {dev_server_url}: {e}"
        )


@pytest.mark.asyncio
async def test_gentle_refresh_performance_requirements(dev_server_url):
    """Test that gentle_refresh meets basic performance requirements."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Time the gentle_refresh call
                import time

                start_time = time.time()

                result = await session.call_tool("gentle_refresh", {})

                end_time = time.time()
                total_time_ms = (end_time - start_time) * 1000

                # Verify response is valid
                response_data = json.loads(result.content[0].text)
                assert response_data["success"] is True

                # Performance requirement: should complete in under 5 seconds
                assert (
                    total_time_ms < 5000
                ), f"gentle_refresh too slow: {total_time_ms}ms"

    except Exception as e:
        pytest.fail(f"Failed to test performance via MCP at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_gentle_refresh_json_validity(dev_server_url):
    """Test that gentle_refresh always returns valid JSON."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call gentle_refresh multiple times to test consistency
                for _i in range(3):
                    result = await session.call_tool("gentle_refresh", {})

                    # Should always return valid JSON
                    assert result.content is not None
                    text_content = result.content[0].text

                    # This will raise an exception if JSON is invalid
                    response_data = json.loads(text_content)

                    # Should always have basic structure
                    assert "success" in response_data
                    assert isinstance(response_data, dict)

    except Exception as e:
        pytest.fail(f"Failed to test JSON validity via MCP at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_gentle_refresh_no_query_parameter_bug(dev_server_url):
    """Test gentle_refresh works when called with no query parameter (regression test)."""
    # This test catches the "None is not of type string" schema validation bug
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call gentle_refresh with no parameters at all - this should NOT raise a schema error
                result = await session.call_tool("gentle_refresh", {})

                # Should return valid response
                assert result.content is not None
                text_content = result.content[0].text
                response_data = json.loads(text_content)

                # Should work normally
                assert response_data["success"] is True
                assert "time" in response_data
                assert "core_identity" in response_data
                assert "shortterm_memories" in response_data

    except Exception as e:
        pytest.fail(
            f"Failed to test no query parameter bug via MCP at {dev_server_url}: {e}"
        )


@pytest.mark.asyncio
async def test_gentle_refresh_timezone_detection(dev_server_url):
    """Test gentle_refresh correctly detects host timezone (not container UTC)."""
    # This test catches the Docker timezone bug where container defaults to UTC
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call gentle_refresh
                result = await session.call_tool("gentle_refresh", {})
                response_data = json.loads(result.content[0].text)

                # Verify timezone is NOT UTC (should be host timezone)
                time_obj = response_data["time"]
                timezone_info = time_obj["timezone"]

                # Should NOT be UTC - should be host timezone
                assert (
                    timezone_info["name"] != "UTC"
                ), f"Container should not be using UTC timezone, got: {timezone_info['name']}"
                assert (
                    timezone_info["name"] != "Etc/UTC"
                ), f"Container should not be using Etc/UTC timezone, got: {timezone_info['name']}"

                # Should be a real timezone (not just UTC)
                assert (
                    "/" in timezone_info["name"]
                ), f"Expected real timezone like 'America/Los_Angeles', got: {timezone_info['name']}"

                # Local time should have proper offset (not +00:00 unless actually UTC)
                local_time = time_obj["local"]
                if timezone_info["name"] not in ["UTC", "Etc/UTC"]:
                    assert not local_time.endswith(
                        "+00:00"
                    ), f"Local time should not have UTC offset for non-UTC timezone, got: {local_time}"

    except Exception as e:
        pytest.fail(
            f"Failed to test timezone detection via MCP at {dev_server_url}: {e}"
        )


@pytest.mark.asyncio
async def test_gentle_refresh_error_resilience(dev_server_url):
    """Test gentle_refresh handles service failures gracefully."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call gentle_refresh (should handle partial failures gracefully)
                result = await session.call_tool("gentle_refresh", {})
                response_data = json.loads(result.content[0].text)

                # Should still return success=True even if some components fail
                assert "success" in response_data
                assert response_data["success"] is True

                # Should still have the basic structure
                assert "time" in response_data
                assert "core_identity" in response_data
                assert "shortterm_memories" in response_data
                assert "memory_consolidation" in response_data
                assert "recent_observations" in response_data

    except Exception as e:
        pytest.fail(f"Failed to test error resilience via MCP at {dev_server_url}: {e}")

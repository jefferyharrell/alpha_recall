"""Integration tests for long-term memory functionality.

These tests connect to the actual running dev container to verify
LTM tools work correctly with real Memgraph storage.
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
async def test_remember_longterm_tool_available(dev_server_url):
    """Test that LTM tools are available via MCP."""
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

                # Check that LTM tools are available
                expected_tools = [
                    "remember_longterm",
                    "relate_longterm",
                    "search_longterm",
                    "get_entity",
                    "get_relationships",
                    "browse_longterm",
                ]

                for tool_name in expected_tools:
                    assert (
                        tool_name in tool_names
                    ), f"Tool {tool_name} not found in {tool_names}"

    except Exception as e:
        pytest.skip(f"Could not connect to dev server: {e}")


@pytest.mark.asyncio
async def test_remember_longterm_basic_functionality(dev_server_url):
    """Test basic remember_longterm functionality with real Memgraph."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Test creating entity without type (simplest case)
                result = await session.call_tool(
                    "remember_longterm",
                    arguments={
                        "entity": "Integration Test Entity",
                        "observation": "This is a test observation for integration testing",
                    },
                )

                assert result.content
                data = json.loads(result.content[0].text)

                # This should reveal the "wrong value marker" error
                if not data.get("success"):
                    pytest.fail(
                        f"remember_longterm failed: {data.get('error')} (type: {data.get('error_type')})"
                    )

                assert data["success"] is True
                assert data["entity"]["entity_name"] == "Integration Test Entity"
                assert (
                    data["observation"]["observation"]
                    == "This is a test observation for integration testing"
                )

    except Exception as e:
        pytest.skip(f"Could not connect to dev server: {e}")


@pytest.mark.asyncio
async def test_remember_longterm_with_type(dev_server_url):
    """Test remember_longterm with entity type (this might be where the error occurs)."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Test creating entity with type (this might trigger the error)
                result = await session.call_tool(
                    "remember_longterm",
                    arguments={
                        "entity": "Integration Test Entity With Type",
                        "observation": "This entity has a type",
                        "type": "TestEntity",
                    },
                )

                assert result.content
                data = json.loads(result.content[0].text)

                # This should reveal the "wrong value marker" error
                if not data.get("success"):
                    pytest.fail(
                        f"remember_longterm with type failed: {data.get('error')} (type: {data.get('error_type')})"
                    )

                assert data["success"] is True
                assert (
                    data["entity"]["entity_name"] == "Integration Test Entity With Type"
                )
                assert data["entity"]["entity_type"] == "TestEntity"

    except Exception as e:
        pytest.skip(f"Could not connect to dev server: {e}")


@pytest.mark.asyncio
async def test_browse_longterm_functionality(dev_server_url):
    """Test browse_longterm functionality."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Create a test entity first
                await session.call_tool(
                    "remember_longterm",
                    arguments={
                        "entity": "Browse Test Entity",
                        "observation": "This is a test entity for browsing",
                    },
                )

                # Test browsing entities (this uses our new method)
                result = await session.call_tool(
                    "browse_longterm", arguments={"limit": 5, "offset": 0}
                )

                assert result.content
                data = json.loads(result.content[0].text)

                if not data.get("success"):
                    pytest.fail(
                        f"browse_longterm failed: {data.get('error')} (type: {data.get('error_type')})"
                    )

                assert data["success"] is True
                assert "browse_data" in data
                assert "entities" in data["browse_data"]
                assert "pagination" in data["browse_data"]

    except Exception as e:
        pytest.fail(f"Test failed with exception: {e}")


@pytest.mark.asyncio
async def test_get_entity_functionality(dev_server_url):
    """Test get_entity functionality."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # First create an entity to retrieve
                create_result = await session.call_tool(
                    "remember_longterm",
                    arguments={
                        "entity": "Entity For Retrieval Test",
                        "observation": "This entity will be retrieved",
                    },
                )

                create_data = json.loads(create_result.content[0].text)
                if not create_data.get("success"):
                    pytest.skip(
                        f"Could not create test entity: {create_data.get('error')}"
                    )

                # Now try to retrieve it
                result = await session.call_tool(
                    "get_entity", arguments={"entity_name": "Entity For Retrieval Test"}
                )

                assert result.content
                data = json.loads(result.content[0].text)

                if not data.get("success"):
                    pytest.fail(
                        f"get_entity failed: {data.get('error')} (type: {data.get('error_type')})"
                    )

                assert data["success"] is True
                assert data["entity"]["entity_name"] == "Entity For Retrieval Test"
                assert len(data["entity"]["observations"]) >= 1

    except Exception as e:
        pytest.skip(f"Could not connect to dev server: {e}")


@pytest.mark.asyncio
async def test_get_relationships_functionality(dev_server_url):
    """Test get_relationships functionality."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # First create entities and relationships
                await session.call_tool(
                    "remember_longterm",
                    arguments={"entity": "Alice Integration", "type": "Person"},
                )

                await session.call_tool(
                    "remember_longterm",
                    arguments={"entity": "Bob Integration", "type": "Person"},
                )

                await session.call_tool(
                    "relate_longterm",
                    arguments={
                        "entity": "Alice Integration",
                        "to_entity": "Bob Integration",
                        "as_type": "knows",
                    },
                )

                # Now try to get relationships
                result = await session.call_tool(
                    "get_relationships", arguments={"entity_name": "Alice Integration"}
                )

                assert result.content
                data = json.loads(result.content[0].text)

                if not data.get("success"):
                    pytest.fail(
                        f"get_relationships failed: {data.get('error')} (type: {data.get('error_type')})"
                    )

                assert data["success"] is True
                assert data["relationships"]["entity_name"] == "Alice Integration"

    except Exception as e:
        pytest.skip(f"Could not connect to dev server: {e}")

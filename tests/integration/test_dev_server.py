"""Integration tests for Alpha-Recall dev server.

These tests connect to the actual running dev container to verify
the MCP server is working correctly. Sits between unit tests and
full e2e tests in our testing pyramid.
"""

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


@pytest.mark.asyncio
async def test_dev_server_health_check(dev_server_url):
    """Test that we can connect to the dev server and call health_check."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call the health_check tool
                result = await session.call_tool("health_check", {})

                # Original MCP returns content in a different format
                assert result.content is not None
                text_content = result.content[0].text

                # Parse the JSON response
                health_data = json.loads(text_content)

                # Check standard health check format
                assert "status" in health_data
                assert health_data["status"] == "ok"
                assert "version" in health_data
                assert health_data["version"] == "1.0.0"
                assert "checks" in health_data
                assert "timestamp" in health_data

                # Check that our dependencies are being monitored
                assert "memgraph" in health_data["checks"]
                assert "redis" in health_data["checks"]

                # Verify timestamp format (should be UTC with +00:00 offset)
                timestamp = health_data["timestamp"]
                assert timestamp.endswith(
                    "+00:00"
                ), f"Timestamp should be UTC format, got: {timestamp}"

    except Exception as e:
        pytest.fail(f"Failed to connect to dev server at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_dev_server_tools_list(dev_server_url):
    """Test that we can list tools from the dev server."""
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

                # Check that health_check tool is available
                assert "health_check" in tool_names

    except Exception as e:
        pytest.fail(f"Failed to list tools from dev server at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_dev_server_connectivity(dev_server_url):
    """Test basic connectivity to the dev server."""
    try:
        async with streamablehttp_client(dev_server_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Basic connectivity test - if we can initialize and list tools, we're connected
                result = await session.list_tools()
                assert len(result.tools) > 0, "Server should have at least one tool"

    except Exception as e:
        pytest.fail(f"Failed to connect to dev server at {dev_server_url}: {e}")

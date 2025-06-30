"""Integration tests for Alpha-Recall dev server.

These tests connect to the actual running dev container to verify
the MCP server is working correctly. Sits between unit tests and
full e2e tests in our testing pyramid.
"""

import json
import pytest
import asyncio
from fastmcp import Client
from src.alpha_recall.config import AlphaRecallSettings


@pytest.fixture(scope="session")
def dev_server_url():
    """Get the dev server URL from configuration."""
    config = AlphaRecallSettings()
    return f"http://localhost:{config.alpha_recall_dev_port}/mcp/"


@pytest.mark.asyncio
async def test_dev_server_health_check(dev_server_url):
    """Test that we can connect to the dev server and call health_check."""
    try:
        async with Client(dev_server_url) as client:
            # Call the health_check tool
            result = await client.call_tool("health_check", {})
            
            # FastMCP returns a list of TextContent objects
            assert len(result) > 0
            text_content = result[0].text
            
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
            
    except Exception as e:
        pytest.fail(f"Failed to connect to dev server at {dev_server_url}: {e}")


@pytest.mark.asyncio
async def test_dev_server_tools_list(dev_server_url):
    """Test that we can list tools from the dev server."""
    try:
        async with Client(dev_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]
            
            # Check that health_check tool is available
            assert "health_check" in tool_names
            
    except Exception as e:
        pytest.fail(f"Failed to list tools from dev server at {dev_server_url}: {e}")


@pytest.mark.asyncio  
async def test_dev_server_connectivity(dev_server_url):
    """Test basic connectivity to the dev server."""
    try:
        async with Client(dev_server_url) as client:
            # Basic ping to test connectivity
            await client.ping()
            
    except Exception as e:
        pytest.fail(f"Failed to ping dev server at {dev_server_url}: {e}")
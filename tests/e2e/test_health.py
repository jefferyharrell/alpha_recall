"""Health check and basic connectivity tests for Alpha-Recall MCP server."""

import json

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_mcp_health_check_tool(test_stack):
    """Test the health_check tool via MCP interface."""
    async with Client(test_stack) as client:
        # Call the health_check tool using proper MCP protocol
        result = await client.call_tool("health_check", {})

        # FastMCP returns a CallToolResult object with content
        assert result.content is not None
        assert len(result.content) > 0
        text_content = result.content[0].text

        # Parse the JSON response
        health_data = json.loads(text_content)

        # Check standard health check format
        assert "status" in health_data
        assert health_data["status"] == "ok"
        assert "version" in health_data
        assert "checks" in health_data
        assert "timestamp" in health_data

        # Check that our dependencies are being monitored
        assert "memgraph" in health_data["checks"]
        assert "redis" in health_data["checks"]


@pytest.mark.asyncio
async def test_full_server_lifecycle(test_stack):
    """Test that we can start, use, and the server cleans up properly."""
    # This test just proves the stack comes up and down cleanly
    # and that our MCP tool works

    async with Client(test_stack) as client:
        # Test basic connectivity
        await client.ping()

        # List available tools
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "health_check" in tool_names

        # Call the health check tool
        result = await client.call_tool("health_check", {})
        assert result.content is not None
        assert len(result.content) > 0

        # Parse the JSON response
        health_data = json.loads(result.content[0].text)
        assert health_data["status"] == "ok"

    # If we get here, the server started and responded
    # The fixture cleanup will test that teardown works

"""End-to-end tests for Alpha-Recall MCP server."""

import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest
from fastmcp import Client


@pytest.fixture(scope="session")
def test_stack():
    """Spin up the full Alpha-Recall test stack."""
    # Path to our test compose file
    compose_file = Path(__file__).parent.parent / "docker" / "e2e.yml"
    project_name = "alpha-recall-e2e-test"

    try:
        # Start the test stack
        print(f"Starting test stack with compose file: {compose_file}")
        print(f"Project name: {project_name}")

        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "-p",
                project_name,
                "up",
                "-d",
                "--build",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        print(f"Docker compose stdout: {result.stdout}")
        if result.stderr:
            print(f"Docker compose stderr: {result.stderr}")

        # Wait for the server to be ready - test actual MCP interface
        server_url = "http://localhost:19006/mcp/"
        max_attempts = 30

        async def check_server():
            async with Client(server_url) as client:
                await client.ping()

        for attempt in range(max_attempts):
            try:
                # Try to initialize the MCP server using proper MCP client
                asyncio.run(check_server())
                print(f"Server ready after {attempt + 1} attempts")
                break
            except Exception:
                pass

            if attempt == max_attempts - 1:
                raise RuntimeError("Test server failed to start within 60 seconds")

            print(f"Attempt {attempt + 1}/{max_attempts} - server not ready yet")
            time.sleep(2)

        yield server_url

    finally:
        # Clean up the test stack
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "-p",
                project_name,
                "down",
                "-v",
                "--remove-orphans",
            ],
            capture_output=True,
        )


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

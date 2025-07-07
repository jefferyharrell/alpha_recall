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
        max_attempts = 90  # Increased timeout for model downloads (3 minutes)

        async def check_server():
            async with Client(server_url) as client:
                await client.ping()

        for attempt in range(max_attempts):
            try:
                # Try to initialize the MCP server using proper MCP client
                asyncio.run(check_server())
                print(f"Server ready after {attempt + 1} attempts")
                break
            except Exception as e:
                # Log the specific error for debugging
                if attempt % 15 == 0:  # Log every 30 seconds
                    print(
                        f"Attempt {attempt + 1}/{max_attempts} - server not ready: {e}"
                    )

            if attempt == max_attempts - 1:
                # Show container logs on failure for debugging
                try:
                    logs_result = subprocess.run(
                        ["docker", "logs", "alpha-recall-test-server"],
                        capture_output=True,
                        text=True,
                    )
                    print(f"Container logs:\n{logs_result.stdout}")
                    if logs_result.stderr:
                        print(f"Container stderr:\n{logs_result.stderr}")
                except Exception:
                    pass
                raise RuntimeError("Test server failed to start within 180 seconds")

            if attempt % 15 == 0 and attempt > 0:  # Progress update every 30 seconds
                print(
                    f"Still waiting... attempt {attempt + 1}/{max_attempts} - allowing time for model downloads"
                )
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


@pytest.mark.asyncio
async def test_remember_shortterm_tool(test_stack):
    """Test the remember_shortterm tool via MCP interface."""
    async with Client(test_stack) as client:
        # Test with short content
        result = await client.call_tool(
            "remember_shortterm",
            {"content": "This is a simple test memory to check performance."},
        )

        # Parse the JSON response
        response_data = json.loads(result.content[0].text)

        # Verify the response structure
        assert "status" in response_data
        assert response_data["status"] == "processed"
        assert "content_length" in response_data
        assert "content_tokens" in response_data
        assert "semantic_embedding_dims" in response_data
        assert "emotional_embedding_dims" in response_data
        assert "timing" in response_data
        assert "performance" in response_data
        assert "correlation_id" in response_data

        # Verify embedding dimensions
        assert response_data["semantic_embedding_dims"] == 768
        assert response_data["emotional_embedding_dims"] == 1024

        # Verify performance metrics are reasonable
        assert response_data["performance"]["semantic_tokens_per_sec"] > 0
        assert response_data["performance"]["emotional_tokens_per_sec"] > 0
        assert response_data["timing"]["total_ms"] > 0

        print(
            f"✅ Short content: {response_data['performance']['total_tokens_per_sec']} tokens/sec"
        )


@pytest.mark.asyncio
async def test_remember_shortterm_performance(test_stack):
    """Test remember_shortterm performance with longer content."""
    async with Client(test_stack) as client:
        long_content = """
        This is a comprehensive test of the remember_shortterm functionality
        with sentence-transformers v5.0.0. We are testing the performance
        improvements that come from upgrading from v2.7.0 to v5.0.0, which
        should provide significant speed improvements for both semantic and
        emotional embedding generation. The goal is to measure real-world
        performance in a containerized environment where we cannot use
        Metal Performance Shaders and are limited to CPU-only processing.
        """

        result = await client.call_tool(
            "remember_shortterm", {"content": long_content.strip()}
        )

        response_data = json.loads(result.content[0].text)

        # Verify faster processing for longer content
        assert response_data["content_tokens"] > 50  # Should be substantial content
        assert (
            response_data["performance"]["total_tokens_per_sec"] > 10
        )  # Better than old baseline

        print(
            f"✅ Long content: {response_data['content_tokens']} tokens in {response_data['timing']['total_ms']}ms"
        )
        print(
            f"📊 Performance: {response_data['performance']['total_tokens_per_sec']} tokens/sec"
        )

        # Verify this is much better than the old ~10 tokens/sec baseline
        assert (
            response_data["performance"]["total_tokens_per_sec"] > 50
        ), "Should be significantly faster than v2.x baseline"


@pytest.mark.asyncio
async def test_remember_shortterm_splash_functionality(test_stack):
    """Test the splash functionality that finds related memories."""
    async with Client(test_stack) as client:
        # Test with content that should trigger our mock related memories
        test_content = "Working on alpha-recall memory embedding improvements"

        result = await client.call_tool("remember_shortterm", {"content": test_content})

        response_data = json.loads(result.content[0].text)

        # Verify basic response structure first
        assert response_data["status"] == "processed"

        # Verify splash section exists
        assert "splash" in response_data, "Response should include splash section"
        splash = response_data["splash"]

        # Verify splash structure
        assert "related_memories_found" in splash
        assert "search_time_ms" in splash
        assert "memories" in splash

        # Verify splash performance
        assert isinstance(splash["related_memories_found"], int)
        assert splash["related_memories_found"] >= 0
        assert isinstance(splash["search_time_ms"], int | float)
        assert splash["search_time_ms"] < 1000, "Splash search should be fast"

        # Verify memories structure
        memories = splash["memories"]
        assert isinstance(memories, list)

        # With our mock implementation and keywords "alpha-recall" and "memory",
        # we should get at least 1 related memory
        assert (
            len(memories) >= 1
        ), "Should find at least one related memory for this content"

        # Verify each memory has the correct structure
        for memory in memories:
            assert "content" in memory
            assert "similarity_score" in memory
            assert "created_at" in memory
            assert "id" in memory
            assert "source" in memory

            # Verify field types and constraints
            assert isinstance(memory["content"], str)
            assert len(memory["content"]) > 0
            assert isinstance(memory["similarity_score"], int | float)
            assert 0.0 <= memory["similarity_score"] <= 1.0
            assert isinstance(memory["created_at"], str)
            assert isinstance(memory["id"], str)
            assert memory["source"] == "mock_splash"

        # Verify memories are sorted by similarity (highest first)
        if len(memories) > 1:
            for i in range(len(memories) - 1):
                assert (
                    memories[i]["similarity_score"]
                    >= memories[i + 1]["similarity_score"]
                ), "Memories should be sorted by similarity score (highest first)"

        print(f"✅ Splash found {len(memories)} related memories")
        print(f"🔍 Search completed in {splash['search_time_ms']}ms")

        if memories:
            top_similarity = memories[0]["similarity_score"]
            print(f"🎯 Top similarity score: {top_similarity}")
            assert (
                top_similarity > 0.5
            ), "Should find strongly related memories for keywords"


@pytest.mark.asyncio
async def test_remember_shortterm_splash_keyword_matching(test_stack):
    """Test that splash finds appropriate memories based on content keywords."""
    async with Client(test_stack) as client:
        # Test different keyword combinations to verify our mock logic
        test_cases = [
            {
                "content": "Testing embedding performance tools",
                "expected_keywords": ["embedding", "tool"],
                "min_memories": 1,
            },
            {
                "content": "Using Claude Code with FastMCP integration",
                "expected_keywords": ["claude code", "fastmcp"],
                "min_memories": 1,
            },
            {
                "content": "Random unrelated content about cooking pasta",
                "expected_keywords": [],
                "min_memories": 1,  # Should still get the general fallback memory
            },
        ]

        for test_case in test_cases:
            result = await client.call_tool(
                "remember_shortterm", {"content": test_case["content"]}
            )

            response_data = json.loads(result.content[0].text)
            memories = response_data["splash"]["memories"]

            assert len(memories) >= test_case["min_memories"], (
                f"Should find at least {test_case['min_memories']} memory(ies) "
                f"for content: {test_case['content']}"
            )

            # Verify that found memories are contextually relevant
            if test_case["expected_keywords"]:
                found_relevant = any(
                    any(
                        keyword.lower() in memory["content"].lower()
                        for keyword in test_case["expected_keywords"]
                    )
                    for memory in memories
                )
                assert found_relevant, (
                    f"Should find memories related to keywords {test_case['expected_keywords']} "
                    f"for content: {test_case['content']}"
                )

            print(f"✅ Keyword test passed for: {test_case['content'][:50]}...")
            print(f"   Found {len(memories)} related memories")

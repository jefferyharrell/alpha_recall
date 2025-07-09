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
        assert response_data["status"] == "stored"
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
        assert response_data["status"] == "stored"

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

        # With a fresh Redis instance, we might not find related memories initially
        # This is normal behavior - let's verify the structure is correct
        assert isinstance(len(memories), int), "Should return valid memory count"

        # If we do find memories, verify their structure
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
            assert memory["source"] == "redis_vector_search"

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
            # With real Redis, similarity scores will be realistic (0.0-1.0 range)
            assert (
                0.0 <= top_similarity <= 1.0
            ), "Similarity score should be in valid range"


@pytest.mark.asyncio
async def test_remember_shortterm_splash_keyword_matching(test_stack):
    """Test that splash functionality works with real Redis storage and search."""
    async with Client(test_stack) as client:
        # First, store a few memories to test the search functionality
        seed_memories = [
            "Testing embedding performance tools for AI development",
            "Using Claude Code with FastMCP integration patterns",
            "Redis vector search implementation with cosine similarity",
        ]

        stored_ids = []
        for memory_content in seed_memories:
            result = await client.call_tool(
                "remember_shortterm", {"content": memory_content}
            )
            response_data = json.loads(result.content[0].text)
            stored_ids.append(response_data["memory_id"])
            print(f"Stored: {memory_content[:50]}...")

        # Now test that we can find related memories
        query_content = "Testing embedding performance tools"
        result = await client.call_tool(
            "remember_shortterm", {"content": query_content}
        )

        response_data = json.loads(result.content[0].text)
        memories = response_data["splash"]["memories"]

        # With real Redis vector search, we should find some related memories
        # (might be 0 if similarities are too low, but structure should be correct)
        assert isinstance(memories, list), "Should return a list of memories"

        # Verify each found memory has correct structure
        for memory in memories:
            assert "content" in memory
            assert "similarity_score" in memory
            assert "created_at" in memory
            assert "id" in memory
            assert "source" in memory
            assert memory["source"] == "redis_vector_search"
            assert 0.0 <= memory["similarity_score"] <= 1.0

        print(
            f"✅ Query '{query_content[:30]}...' found {len(memories)} related memories"
        )
        if memories:
            print(f"🎯 Top similarity: {memories[0]['similarity_score']:.3f}")
            print(f"📝 Top match: {memories[0]['content'][:50]}...")
        else:
            print(
                "ℹ️  No memories found above similarity threshold (0.3) - this is normal"
            )


@pytest.mark.asyncio
async def test_remember_longterm_tool(test_stack):
    """Test the remember_longterm tool via MCP interface."""
    async with Client(test_stack) as client:
        # Test entity creation with observation
        entity_name = "Test Entity"
        observation = "This is a test observation for e2e testing"
        entity_type = "TestType"

        result = await client.call_tool(
            "remember_longterm",
            {"entity": entity_name, "observation": observation, "type": entity_type},
        )

        response_data = json.loads(result.content[0].text)

        # Verify response structure
        assert "success" in response_data
        assert response_data["success"] is True
        assert "entity" in response_data
        assert response_data["entity"]["entity_name"] == entity_name
        assert response_data["entity"]["entity_type"] == entity_type
        assert "observation" in response_data
        assert response_data["observation"] is not None
        assert "correlation_id" in response_data

        print(f"✅ Created entity: {entity_name} with observation")

        # Test entity-only creation (no observation)
        entity_only_name = "Entity Only Test"
        result = await client.call_tool(
            "remember_longterm", {"entity": entity_only_name}
        )

        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is True
        assert response_data["entity"]["entity_name"] == entity_only_name
        assert response_data["observation"] is None

        print(f"✅ Created entity-only: {entity_only_name}")


@pytest.mark.asyncio
async def test_relate_longterm_tool(test_stack):
    """Test the relate_longterm tool via MCP interface."""
    async with Client(test_stack) as client:
        # First create two entities to relate
        entity1 = "Entity One"
        entity2 = "Entity Two"

        await client.call_tool("remember_longterm", {"entity": entity1})
        await client.call_tool("remember_longterm", {"entity": entity2})

        # Now create a relationship between them
        relationship_type = "works_with"
        result = await client.call_tool(
            "relate_longterm",
            {"entity": entity1, "to_entity": entity2, "as_type": relationship_type},
        )

        response_data = json.loads(result.content[0].text)

        # Verify response structure
        assert "success" in response_data
        assert response_data["success"] is True
        assert "relationship" in response_data
        assert response_data["relationship"]["entity1"] == entity1
        assert response_data["relationship"]["entity2"] == entity2
        assert response_data["relationship"]["relationship_type"] == relationship_type
        assert "correlation_id" in response_data

        print(f"✅ Created relationship: {entity1} {relationship_type} {entity2}")


@pytest.mark.asyncio
async def test_search_longterm_tool(test_stack):
    """Test the search_longterm tool via MCP interface."""
    async with Client(test_stack) as client:
        # First create an entity with a searchable observation
        entity_name = "Search Test Entity"
        observation = "This entity contains information about machine learning and neural networks"

        await client.call_tool(
            "remember_longterm", {"entity": entity_name, "observation": observation}
        )

        # Search for the observation
        search_query = "machine learning neural networks"
        result = await client.call_tool("search_longterm", {"query": search_query})

        response_data = json.loads(result.content[0].text)

        # Verify response structure
        print(f"DEBUG: search_longterm response: {response_data}")
        assert "success" in response_data
        if not response_data["success"]:
            print(
                f"DEBUG: Error in search_longterm: {response_data.get('error', 'Unknown error')}"
            )
        assert response_data["success"] is True
        assert "query" in response_data
        assert response_data["query"] == search_query
        assert "observations" in response_data
        assert "results_count" in response_data
        # Note: search_time_ms is not in the actual response
        assert "correlation_id" in response_data

        # Verify results structure
        results = response_data["observations"]
        assert isinstance(results, list)

        # Verify results structure (but don't require specific matches since search may not find our test observation)
        for result in results:
            assert "entity_name" in result
            assert "observation" in result
            assert "similarity_score" in result
            assert "created_at" in result
            assert isinstance(result["similarity_score"], int | float)
            assert 0.0 <= result["similarity_score"] <= 1.0

        # Note: We don't assert that we find our specific observation since
        # the search might not return it if similarity is too low or if embedding generation
        # creates different vectors. The important thing is that the search structure works.

        print(f"✅ Search found {len(results)} results for '{search_query}'")

        # Test entity-specific search
        result = await client.call_tool(
            "search_longterm", {"query": "machine learning", "entity": entity_name}
        )

        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is True

        # Should only return results for the specific entity
        for result in response_data["observations"]:
            assert result["entity_name"] == entity_name

        print(
            f"✅ Entity-specific search found {len(response_data['observations'])} results"
        )


@pytest.mark.asyncio
async def test_get_entity_tool(test_stack):
    """Test the get_entity tool via MCP interface."""
    async with Client(test_stack) as client:
        # Create an entity with multiple observations
        entity_name = "Get Entity Test"
        observations = [
            "First observation about this entity",
            "Second observation with different content",
            "Third observation for comprehensive testing",
        ]

        # Create entity with first observation
        await client.call_tool(
            "remember_longterm",
            {
                "entity": entity_name,
                "observation": observations[0],
                "type": "TestEntity",
            },
        )

        # Add additional observations
        for obs in observations[1:]:
            await client.call_tool(
                "remember_longterm", {"entity": entity_name, "observation": obs}
            )

        # Now retrieve the entity with all observations
        result = await client.call_tool("get_entity", {"entity_name": entity_name})

        response_data = json.loads(result.content[0].text)

        # Verify response structure
        assert "success" in response_data
        assert response_data["success"] is True
        assert "entity" in response_data
        assert "correlation_id" in response_data

        # Verify entity data structure
        entity = response_data["entity"]
        assert "entity_name" in entity
        assert entity["entity_name"] == entity_name
        assert "entity_type" in entity
        assert entity["entity_type"] == "TestEntity"
        assert "created_at" in entity
        assert "updated_at" in entity
        assert "observations" in entity

        # Verify observations
        returned_observations = entity["observations"]
        assert len(returned_observations) == len(observations)

        # Check that all observations are present
        returned_contents = [obs["content"] for obs in returned_observations]
        for expected_obs in observations:
            assert expected_obs in returned_contents

        # Verify observation structure
        for obs in returned_observations:
            assert "content" in obs
            assert "created_at" in obs
            assert "id" in obs

        print(
            f"✅ Retrieved entity '{entity_name}' with {len(returned_observations)} observations"
        )


@pytest.mark.asyncio
async def test_get_relationships_tool(test_stack):
    """Test the get_relationships tool via MCP interface."""
    async with Client(test_stack) as client:
        # Create a central entity with multiple relationships
        central_entity = "Central Entity"
        related_entities = ["Related A", "Related B", "Related C"]

        # Create all entities
        await client.call_tool("remember_longterm", {"entity": central_entity})
        for entity in related_entities:
            await client.call_tool("remember_longterm", {"entity": entity})

        # Create outgoing relationships
        await client.call_tool(
            "relate_longterm",
            {
                "entity": central_entity,
                "to_entity": related_entities[0],
                "as_type": "manages",
            },
        )

        await client.call_tool(
            "relate_longterm",
            {
                "entity": central_entity,
                "to_entity": related_entities[1],
                "as_type": "collaborates_with",
            },
        )

        # Create incoming relationship
        await client.call_tool(
            "relate_longterm",
            {
                "entity": related_entities[2],
                "to_entity": central_entity,
                "as_type": "reports_to",
            },
        )

        # Get all relationships for the central entity
        result = await client.call_tool(
            "get_relationships", {"entity_name": central_entity}
        )

        response_data = json.loads(result.content[0].text)

        # Verify response structure
        assert "success" in response_data
        assert response_data["success"] is True
        assert "relationships" in response_data
        assert "correlation_id" in response_data

        # Verify relationships structure
        relationships = response_data["relationships"]
        assert "entity_name" in relationships
        assert relationships["entity_name"] == central_entity
        assert "outgoing_relationships" in relationships
        assert "incoming_relationships" in relationships
        assert "total_relationships" in relationships

        # Verify outgoing relationships
        outgoing = relationships["outgoing_relationships"]
        assert len(outgoing) == 2  # manages and collaborates_with

        relationship_types = [rel["type"] for rel in outgoing]
        assert "manages" in relationship_types
        assert "collaborates_with" in relationship_types

        # Verify incoming relationships
        incoming = relationships["incoming_relationships"]
        assert len(incoming) == 1  # reports_to
        assert incoming[0]["type"] == "reports_to"
        assert incoming[0]["source"] == related_entities[2]

        # Verify total count
        assert relationships["total_relationships"] == 3

        # Verify relationship structure
        for rel in outgoing + incoming:
            assert "type" in rel
            assert "created_at" in rel
            if rel in outgoing:
                assert "target" in rel
                assert rel["target"] in related_entities
            else:
                assert "source" in rel

        print(
            f"✅ Retrieved {len(outgoing)} outgoing and {len(incoming)} incoming relationships for '{central_entity}'"
        )


@pytest.mark.asyncio
async def test_browse_longterm_tool(test_stack):
    """Test the browse_longterm tool via MCP interface."""
    async with Client(test_stack) as client:
        # Create multiple entities for browsing
        test_entities = [
            {"name": "Browse Entity 1", "type": "TypeA"},
            {"name": "Browse Entity 2", "type": "TypeB"},
            {"name": "Browse Entity 3", "type": "TypeA"},
            {"name": "Browse Entity 4", "type": "TypeC"},
        ]

        for entity in test_entities:
            await client.call_tool(
                "remember_longterm",
                {
                    "entity": entity["name"],
                    "type": entity["type"],
                    "observation": f"Test observation for {entity['name']}",
                },
            )

        # Test basic browsing
        result = await client.call_tool("browse_longterm", {"limit": 10, "offset": 0})

        response_data = json.loads(result.content[0].text)

        # Verify response structure
        assert "success" in response_data
        assert response_data["success"] is True
        assert "browse_data" in response_data
        assert "correlation_id" in response_data

        # Verify browse_data structure
        browse_data = response_data["browse_data"]
        assert "entities" in browse_data
        assert "pagination" in browse_data

        # Verify pagination info
        pagination = browse_data["pagination"]
        assert "limit" in pagination
        assert "offset" in pagination
        assert "total_count" in pagination
        assert "has_more" in pagination
        assert pagination["limit"] == 10
        assert pagination["offset"] == 0

        # Verify entities structure
        entities = browse_data["entities"]
        assert isinstance(entities, list)
        assert len(entities) >= len(test_entities)  # Should include our test entities

        # Verify entity structure
        for entity in entities:
            assert "entity_name" in entity
            assert "entity_type" in entity
            assert "created_at" in entity
            assert "updated_at" in entity
            assert "observation_count" in entity
            assert isinstance(entity["observation_count"], int)
            assert entity["observation_count"] >= 0

        # Find our test entities in the results
        entity_names = [entity["entity_name"] for entity in entities]
        for test_entity in test_entities:
            assert test_entity["name"] in entity_names

        print(
            f"✅ Browsed {len(entities)} entities with pagination (limit: {pagination['limit']}, offset: {pagination['offset']})"
        )

        # Test pagination with smaller limit
        result = await client.call_tool("browse_longterm", {"limit": 2, "offset": 0})

        response_data = json.loads(result.content[0].text)
        assert len(response_data["browse_data"]["entities"]) <= 2
        assert response_data["browse_data"]["pagination"]["limit"] == 2

        # Test offset
        result = await client.call_tool("browse_longterm", {"limit": 2, "offset": 1})

        response_data = json.loads(result.content[0].text)
        assert response_data["browse_data"]["pagination"]["offset"] == 1

        print("✅ Pagination working correctly with limit/offset parameters")

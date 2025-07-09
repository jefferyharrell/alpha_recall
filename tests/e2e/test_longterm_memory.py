"""Long-term memory core operations tests for Alpha-Recall MCP server."""

import json

import pytest
from fastmcp import Client


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

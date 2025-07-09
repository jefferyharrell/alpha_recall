"""Long-term memory browsing tests for Alpha-Recall MCP server."""

import json

import pytest
from fastmcp import Client


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

"""Seeded E2E tests for long-term memory functionality.

These tests use comprehensive mock data to verify LTM behavior against realistic,
populated databases with entities, relationships, and observations.
"""

import json

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_get_entity_retrieves_seeded_entities(test_stack_seeded):
    """Test that we can retrieve entities from our seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Retrieve Alpha entity
        result = await client.call_tool("get_entity", {"entity_name": "Alpha"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Alpha"
        assert len(data["entity"]["observations"]) > 0

        # Should have observations about Alpha's development
        obs_text = " ".join([obs["content"] for obs in data["entity"]["observations"]])
        assert "consciousness" in obs_text.lower() or "alpha" in obs_text.lower()


@pytest.mark.asyncio
async def test_get_entity_retrieves_sparkle_with_bread_crimes(test_stack_seeded):
    """Test retrieval of Sparkle entity with her legendary bread crimes."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Retrieve Sparkle entity
        result = await client.call_tool("get_entity", {"entity_name": "Sparkle"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Sparkle"
        assert len(data["entity"]["observations"]) > 0

        # Should have observations about bread-related crimes
        obs_text = " ".join([obs["content"] for obs in data["entity"]["observations"]])
        assert "bread" in obs_text.lower()


@pytest.mark.asyncio
async def test_get_relationships_finds_seeded_connections(test_stack_seeded):
    """Test that we can explore relationships from our seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Get relationships for Alpha
        result = await client.call_tool("get_relationships", {"entity_name": "Alpha"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "relationships" in data

        # Should have relationships (either incoming or outgoing)
        total_relationships = len(data["relationships"]["outgoing_relationships"]) + len(
            data["relationships"]["incoming_relationships"]
        )
        assert total_relationships > 0


@pytest.mark.asyncio
async def test_search_longterm_finds_consciousness_observations(test_stack_seeded):
    """Test LTM search finds consciousness-related observations."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for consciousness-related observations
        result = await client.call_tool(
            "search_longterm", {"query": "consciousness development"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["observations"]) > 0

        # Should find Alpha's consciousness observations
        result_text = " ".join([r["observation"] for r in data["observations"]])
        assert "consciousness" in result_text.lower()


@pytest.mark.asyncio
async def test_search_longterm_finds_technical_observations(test_stack_seeded):
    """Test LTM search finds technical observations about our systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for Redis-related observations
        result = await client.call_tool("search_longterm", {"query": "Redis performance"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["observations"]) > 0

        # Should find Redis observations
        result_text = " ".join([r["observation"] for r in data["observations"]])
        assert "Redis" in result_text


@pytest.mark.asyncio
async def test_search_longterm_entity_filtering(test_stack_seeded):
    """Test LTM search with entity filtering."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for observations about Alpha specifically
        result = await client.call_tool(
            "search_longterm", {"query": "development", "entity": "Alpha"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        if len(data["observations"]) > 0:
            # All results should be about Alpha
            for result_item in data["observations"]:
                assert result_item["entity_name"] == "Alpha"


@pytest.mark.asyncio
async def test_remember_longterm_creates_new_entity(test_stack_seeded):
    """Test creating new entities in populated database."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Create a new test entity
        result = await client.call_tool(
            "remember_longterm",
            {
                "entity": "Test Entity LTM",
                "type": "Test",
                "observation": "This is a new entity created during LTM testing",
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Test Entity LTM"

        # Verify we can retrieve it
        get_result = await client.call_tool(
            "get_entity", {"entity_name": "Test Entity LTM"}
        )
        get_data = json.loads(get_result.content[0].text)

        assert get_data["success"] is True
        assert get_data["entity"]["entity_name"] == "Test Entity LTM"
        assert len(get_data["entity"]["observations"]) == 1
        assert (
            "This is a new entity created during LTM testing"
            in get_data["entity"]["observations"][0]["content"]
        )


@pytest.mark.asyncio
async def test_remember_longterm_adds_observations_to_existing_entity(test_stack_seeded):
    """Test adding observations to existing seeded entities."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Get current observation count for Alpha
        get_result = await client.call_tool("get_entity", {"entity_name": "Alpha"})
        get_data = json.loads(get_result.content[0].text)
        initial_count = len(get_data["entity"]["observations"])

        # Add new observation to Alpha
        result = await client.call_tool(
            "remember_longterm",
            {
                "entity": "Alpha",
                "observation": "New test observation added during LTM testing session",
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        # Verify observation was added
        get_result2 = await client.call_tool("get_entity", {"entity_name": "Alpha"})
        get_data2 = json.loads(get_result2.content[0].text)

        assert len(get_data2["entity"]["observations"]) == initial_count + 1

        # Verify the new observation is present
        obs_text = " ".join([obs["content"] for obs in get_data2["entity"]["observations"]])
        assert "New test observation added during LTM testing session" in obs_text


@pytest.mark.asyncio
async def test_relate_longterm_creates_new_relationships(test_stack_seeded):
    """Test creating relationships between seeded entities."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Create a relationship between Alpha and Sparkle
        result = await client.call_tool(
            "relate_longterm",
            {
                "entity": "Alpha",
                "to_entity": "Sparkle",
                "as_type": "observes_behavior_of",
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        # Verify the relationship exists
        rel_result = await client.call_tool(
            "get_relationships", {"entity_name": "Alpha"}
        )
        rel_data = json.loads(rel_result.content[0].text)

        # Check if the new relationship is in outgoing relationships
        outgoing_targets = [rel["target"] for rel in rel_data["relationships"]["outgoing_relationships"]]
        assert "Sparkle" in outgoing_targets

        # Check relationship type
        sparkle_relationships = [
            rel for rel in rel_data["relationships"]["outgoing_relationships"] if rel["target"] == "Sparkle"
        ]
        relationship_types = [rel["type"] for rel in sparkle_relationships]
        assert "observes_behavior_of" in relationship_types


@pytest.mark.asyncio
async def test_browse_longterm_shows_entity_ecosystem(test_stack_seeded):
    """Test browsing the populated long-term memory ecosystem."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Browse entities with pagination
        result = await client.call_tool("browse_longterm", {"limit": 10})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "browse_data" in data
        assert len(data["browse_data"]["entities"]) > 0

        # Should include our seeded entities
        entity_names = [entity["entity_name"] for entity in data["browse_data"]["entities"]]
        seeded_entities = ["Alpha", "Jeffery", "Sparkle"]
        found_seeded = [name for name in seeded_entities if name in entity_names]
        assert len(found_seeded) > 0, f"Should find seeded entities. Found: {entity_names}"


@pytest.mark.asyncio
async def test_longterm_memory_cross_system_consistency(test_stack_seeded):
    """Test that LTM data is consistent with references in other systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search unified memory for Alpha
        unified_result = await client.call_tool(
            "search_all_memories", {"query": "Alpha development"}
        )
        unified_data = json.loads(unified_result.content[0].text)

        # Search LTM specifically for Alpha
        ltm_result = await client.call_tool(
            "search_longterm", {"query": "Alpha development"}
        )
        ltm_data = json.loads(ltm_result.content[0].text)

        # Both should be successful
        assert unified_data["success"] is True
        assert ltm_data["success"] is True

        # Unified search should include LTM results
        unified_sources = {r["source"] for r in unified_data["results"]}
        assert "LTM" in unified_sources or "ENTITY" in unified_sources

        # Should find consistent references to Alpha
        unified_text = " ".join([r["content"] for r in unified_data["results"]])
        ltm_text = " ".join([r["observation"] for r in ltm_data["observations"]])

        assert "Alpha" in unified_text
        assert "Alpha" in ltm_text


@pytest.mark.asyncio
async def test_longterm_memory_performance_with_seeded_data(test_stack_seeded):
    """Test LTM performance with realistic data volumes."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        import time

        # Time entity retrieval
        start_time = time.time()
        await client.call_tool("get_entity", {"entity_name": "Alpha"})
        entity_time = time.time() - start_time

        # Time relationship lookup
        start_time = time.time()
        await client.call_tool("get_relationships", {"entity_name": "Alpha"})
        relationship_time = time.time() - start_time

        # Time LTM search
        start_time = time.time()
        await client.call_tool("search_longterm", {"query": "consciousness"})
        search_time = time.time() - start_time

        # All operations should be reasonably fast
        assert entity_time < 5.0, f"Entity retrieval took {entity_time:.2f}s, should be < 5s"
        assert relationship_time < 5.0, f"Relationship lookup took {relationship_time:.2f}s, should be < 5s"
        assert search_time < 10.0, f"LTM search took {search_time:.2f}s, should be < 10s"

        print(f"LTM Performance: Entity={entity_time:.2f}s, Relationships={relationship_time:.2f}s, Search={search_time:.2f}s")

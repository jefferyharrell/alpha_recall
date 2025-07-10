"""Seeded E2E tests for long-term memory functionality.

These tests use comprehensive mock data to verify LTM behavior against realistic,
populated databases with entities, relationships, and observations.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_get_entity_retrieves_seeded_entities(test_stack_seeded):
    """Test that we can retrieve entities from our seeded data.

    Since models are already warmed up, this should be fast.
    """
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Retrieve Alpha entity
        result = await time_mcp_call(client, "get_entity", {"entity_name": "Alpha"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Alpha"
        assert len(data["entity"]["observations"]) > 0

        # Should have observations about Alpha's development
        obs_text = " ".join([obs["content"] for obs in data["entity"]["observations"]])
        assert "consciousness" in obs_text.lower() or "alpha" in obs_text.lower()

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_get_entity":
                latest_duration = metric["duration_ms"]
                break

        assert latest_duration is not None, "Should have recorded timing for get_entity"
        assert (
            latest_duration < 500
        ), f"get_entity took {latest_duration:.1f}ms, should be <500ms with warm models"

        print(f"🔍 get_entity completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_get_entity_retrieves_sparkle_with_bread_crimes(test_stack_seeded):
    """Test retrieval of Sparkle entity with her legendary bread crimes."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Retrieve Sparkle entity
        result = await time_mcp_call(client, "get_entity", {"entity_name": "Sparkle"})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Sparkle"
        assert len(data["entity"]["observations"]) > 0

        # Should have observations about bread-related crimes
        obs_text = " ".join([obs["content"] for obs in data["entity"]["observations"]])
        assert "bread" in obs_text.lower()

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_get_entity":
                latest_duration = metric["duration_ms"]
                break

        assert latest_duration is not None, "Should have recorded timing for get_entity"
        assert (
            latest_duration < 500
        ), f"get_entity took {latest_duration:.1f}ms, should be <500ms"

        print(f"🍞 Sparkle's crimes retrieved in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_get_relationships_finds_seeded_connections(test_stack_seeded):
    """Test that we can explore relationships from our seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Get relationships for Alpha
        result = await time_mcp_call(
            client, "get_relationships", {"entity_name": "Alpha"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "relationships" in data

        # Should have relationships (either incoming or outgoing)
        total_relationships = len(
            data["relationships"]["outgoing_relationships"]
        ) + len(data["relationships"]["incoming_relationships"])
        assert total_relationships > 0

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_get_relationships":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for get_relationships"
        assert (
            latest_duration < 500
        ), f"get_relationships took {latest_duration:.1f}ms, should be <500ms"

        print(f"🔗 Relationships retrieved in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_longterm_finds_consciousness_observations(test_stack_seeded):
    """Test LTM search finds consciousness-related observations."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for consciousness-related observations
        result = await time_mcp_call(
            client, "search_longterm", {"query": "consciousness development"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["observations"]) > 0

        # Should find Alpha's consciousness observations
        result_text = " ".join([r["observation"] for r in data["observations"]])
        assert "consciousness" in result_text.lower()

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_longterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_longterm"
        assert (
            latest_duration < 500
        ), f"search_longterm took {latest_duration:.1f}ms, should be <500ms with warm models"

        print(f"🧠 Consciousness search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_longterm_finds_technical_observations(test_stack_seeded):
    """Test LTM search finds technical observations about our systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for Redis-related observations
        result = await time_mcp_call(
            client, "search_longterm", {"query": "Redis performance"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert len(data["observations"]) > 0

        # Should find Redis observations
        result_text = " ".join([r["observation"] for r in data["observations"]])
        assert "Redis" in result_text

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_longterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_longterm"
        assert (
            latest_duration < 500
        ), f"search_longterm took {latest_duration:.1f}ms, should be <500ms"

        print(f"⚡ Redis search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_longterm_entity_filtering(test_stack_seeded):
    """Test LTM search with entity filtering."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for observations about Alpha specifically
        result = await time_mcp_call(
            client, "search_longterm", {"query": "development", "entity": "Alpha"}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        if len(data["observations"]) > 0:
            # All results should be about Alpha
            for result_item in data["observations"]:
                assert result_item["entity_name"] == "Alpha"

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_longterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_longterm"
        assert (
            latest_duration < 300
        ), f"search_longterm took {latest_duration:.1f}ms, should be <300ms"

        print(f"🎯 Entity-filtered search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_remember_longterm_creates_new_entity(test_stack_seeded):
    """Test creating new entities in populated database."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Create a new test entity
        result = await time_mcp_call(
            client,
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

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_remember_longterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for remember_longterm"
        assert (
            latest_duration < 300
        ), f"remember_longterm took {latest_duration:.1f}ms, should be <300ms"

        print(f"💾 Entity creation completed in {latest_duration:.1f}ms")

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
@performance_test
async def test_remember_longterm_adds_observations_to_existing_entity(
    test_stack_seeded,
):
    """Test adding observations to existing seeded entities."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Get current observation count for Alpha
        get_result = await client.call_tool("get_entity", {"entity_name": "Alpha"})
        get_data = json.loads(get_result.content[0].text)
        initial_count = len(get_data["entity"]["observations"])

        # Add new observation to Alpha
        result = await time_mcp_call(
            client,
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

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_remember_longterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for remember_longterm"
        assert (
            latest_duration < 300
        ), f"remember_longterm took {latest_duration:.1f}ms, should be <300ms"

        print(f"📝 Observation added in {latest_duration:.1f}ms")

        # Verify the new observation is present
        obs_text = " ".join(
            [obs["content"] for obs in get_data2["entity"]["observations"]]
        )
        assert "New test observation added during LTM testing session" in obs_text


@pytest.mark.asyncio
@performance_test
async def test_relate_longterm_creates_new_relationships(test_stack_seeded):
    """Test creating relationships between seeded entities."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Create a relationship between Alpha and Sparkle
        result = await time_mcp_call(
            client,
            "relate_longterm",
            {
                "entity": "Alpha",
                "to_entity": "Sparkle",
                "as_type": "observes_behavior_of",
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_relate_longterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for relate_longterm"
        assert (
            latest_duration < 100
        ), f"relate_longterm took {latest_duration:.1f}ms, should be <100ms"

        print(f"🔗 Relationship created in {latest_duration:.1f}ms")

        # Verify the relationship exists
        rel_result = await client.call_tool(
            "get_relationships", {"entity_name": "Alpha"}
        )
        rel_data = json.loads(rel_result.content[0].text)

        # Check if the new relationship is in outgoing relationships
        outgoing_targets = [
            rel["target"] for rel in rel_data["relationships"]["outgoing_relationships"]
        ]
        assert "Sparkle" in outgoing_targets

        # Check relationship type
        sparkle_relationships = [
            rel
            for rel in rel_data["relationships"]["outgoing_relationships"]
            if rel["target"] == "Sparkle"
        ]
        relationship_types = [rel["type"] for rel in sparkle_relationships]
        assert "observes_behavior_of" in relationship_types


@pytest.mark.asyncio
@performance_test
async def test_browse_longterm_shows_entity_ecosystem(test_stack_seeded):
    """Test browsing the populated long-term memory ecosystem."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Browse entities with pagination
        result = await time_mcp_call(client, "browse_longterm", {"limit": 10})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "browse_data" in data
        assert len(data["browse_data"]["entities"]) > 0

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_browse_longterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for browse_longterm"
        assert (
            latest_duration < 100
        ), f"browse_longterm took {latest_duration:.1f}ms, should be <100ms"

        print(f"📊 Browse completed in {latest_duration:.1f}ms")

        # Should include our seeded entities
        entity_names = [
            entity["entity_name"] for entity in data["browse_data"]["entities"]
        ]
        seeded_entities = ["Alpha", "Jeffery", "Sparkle"]
        found_seeded = [name for name in seeded_entities if name in entity_names]
        assert (
            len(found_seeded) > 0
        ), f"Should find seeded entities. Found: {entity_names}"


@pytest.mark.asyncio
@performance_test
async def test_longterm_memory_cross_system_consistency(test_stack_seeded):
    """Test that LTM data is consistent with references in other systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search unified memory for Alpha
        unified_result = await time_mcp_call(
            client, "search_all_memories", {"query": "Alpha development"}
        )
        unified_data = json.loads(unified_result.content[0].text)

        # Search LTM specifically for Alpha
        ltm_result = await time_mcp_call(
            client, "search_longterm", {"query": "Alpha development"}
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

        # Assert fast performance for both searches
        from tests.e2e.fixtures.performance import collector

        # Check unified search timing
        unified_duration = None
        ltm_duration = None
        for metric in reversed(collector.get_metrics()):
            if (
                metric["operation"] == "mcp_call_search_all_memories"
                and unified_duration is None
            ):
                unified_duration = metric["duration_ms"]
            elif (
                metric["operation"] == "mcp_call_search_longterm"
                and ltm_duration is None
            ):
                ltm_duration = metric["duration_ms"]
            if unified_duration is not None and ltm_duration is not None:
                break

        assert (
            unified_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            ltm_duration is not None
        ), "Should have recorded timing for search_longterm"
        assert (
            unified_duration < 500
        ), f"search_all_memories took {unified_duration:.1f}ms, should be <500ms"
        assert (
            ltm_duration < 300
        ), f"search_longterm took {ltm_duration:.1f}ms, should be <300ms"

        print(
            f"🔄 Cross-system search: unified {unified_duration:.1f}ms, LTM {ltm_duration:.1f}ms"
        )


@pytest.mark.asyncio
@performance_test
async def test_longterm_memory_performance_with_seeded_data(test_stack_seeded):
    """Test LTM performance with realistic data volumes using our instrumentation."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Test entity retrieval with instrumentation
        entity_result = await time_mcp_call(
            client, "get_entity", {"entity_name": "Alpha"}
        )

        # Test relationship lookup with instrumentation
        relationship_result = await time_mcp_call(
            client, "get_relationships", {"entity_name": "Alpha"}
        )

        # Test LTM search with instrumentation
        search_result = await time_mcp_call(
            client, "search_longterm", {"query": "consciousness"}
        )

        # Verify all operations succeeded
        assert json.loads(entity_result.content[0].text)["success"] is True
        assert json.loads(relationship_result.content[0].text)["success"] is True
        assert json.loads(search_result.content[0].text)["success"] is True

        # Assert performance with our refined expectations
        from tests.e2e.fixtures.performance import collector

        metrics = collector.get_metrics()
        recent_metrics = metrics[-3:]  # Get the 3 most recent metrics

        for metric in recent_metrics:
            operation = metric["operation"]
            duration = metric["duration_ms"]

            if operation == "mcp_call_get_entity":
                assert (
                    duration < 100
                ), f"get_entity took {duration:.1f}ms, should be <100ms"
            elif operation == "mcp_call_get_relationships":
                assert (
                    duration < 100
                ), f"get_relationships took {duration:.1f}ms, should be <100ms"
            elif operation == "mcp_call_search_longterm":
                assert (
                    duration < 300
                ), f"search_longterm took {duration:.1f}ms, should be <300ms"

        print(
            "🎯 Performance test completed - all operations within expected thresholds"
        )

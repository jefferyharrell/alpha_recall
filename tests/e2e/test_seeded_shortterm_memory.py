"""Seeded E2E tests for short-term memory functionality.

These tests use comprehensive mock data to verify STM behavior against realistic,
populated databases with temporal data, search patterns, and TTL expiration.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_remember_shortterm_stores_memory_with_embeddings(test_stack_seeded):
    """Test that STM stores memories with both semantic and emotional embeddings."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Store a new memory with rich content
        result = await time_mcp_call(
            client,
            "remember_shortterm",
            {
                "content": "Alpha discovered an elegant solution to the embedding performance optimization challenge today"
            },
        )
        data = json.loads(result.content[0].text)

        assert data["status"] == "stored"
        assert "memory_id" in data
        assert data["content_tokens"] > 0
        assert data["semantic_embedding_dims"] == 768
        assert data["emotional_embedding_dims"] == 1024

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_remember_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for remember_shortterm"
        assert (
            latest_duration < 300
        ), f"remember_shortterm took {latest_duration:.1f}ms, should be <300ms"

        print(f"💾 STM storage completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_browse_shortterm_with_seeded_data(test_stack_seeded):
    """Test browsing STM with realistic seeded data volumes."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Browse recent STM entries
        result = await time_mcp_call(
            client, "browse_shortterm", {"limit": 10, "order": "desc"}
        )
        data = json.loads(result.content[0].text)

        assert "memories" in data
        assert len(data["memories"]) > 0
        assert "pagination" in data
        assert data["pagination"]["total_in_range"] >= len(data["memories"])

        # Should include seeded STM data
        memory_contents = [mem["content"] for mem in data["memories"]]
        assert any(
            "Alpha" in content for content in memory_contents
        ), "Should find Alpha-related memories"

        # Assert fast performance (browse should be graph-only)
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_browse_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for browse_shortterm"
        assert (
            latest_duration < 100
        ), f"browse_shortterm took {latest_duration:.1f}ms, should be <100ms"

        print(f"📋 STM browse completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_browse_shortterm_temporal_filtering(test_stack_seeded):
    """Test STM browsing with temporal filters."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Browse with time window filter
        result = await time_mcp_call(
            client, "browse_shortterm", {"limit": 20, "since": "1h"}
        )
        data = json.loads(result.content[0].text)

        # browse_shortterm returns data directly, no "success" field
        assert "memories" in data

        # All returned memories should be within the time window
        if len(data["memories"]) > 0:
            # Check that memories have recent timestamps
            # (Exact validation would require comparing against current time)
            for memory in data["memories"]:
                assert "created_at" in memory
                assert memory["created_at"] is not None

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_browse_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for browse_shortterm"
        assert (
            latest_duration < 100
        ), f"browse_shortterm took {latest_duration:.1f}ms, should be <100ms"

        print(f"⏰ Temporal browse completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_shortterm_semantic_with_seeded_data(test_stack_seeded):
    """Test STM semantic search against seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for Alpha-related memories
        result = await time_mcp_call(
            client,
            "search_shortterm",
            {
                "query": "Alpha development consciousness",
                "search_type": "semantic",
                "limit": 10,
            },
        )
        data = json.loads(result.content[0].text)

        # search_shortterm returns data directly, no "success" field
        assert "memories" in data
        assert len(data["memories"]) > 0

        # Should find Alpha-related content
        result_text = " ".join([mem["content"] for mem in data["memories"]])
        assert "Alpha" in result_text

        # Memories should have similarity scores
        for memory in data["memories"]:
            assert "similarity_score" in memory
            assert 0.0 <= memory["similarity_score"] <= 1.0

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_shortterm"
        assert (
            latest_duration < 300
        ), f"search_shortterm took {latest_duration:.1f}ms, should be <300ms"

        print(f"🔍 STM semantic search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_shortterm_emotional_with_seeded_data(test_stack_seeded):
    """Test STM emotional search against seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for emotionally similar content
        result = await time_mcp_call(
            client,
            "search_shortterm",
            {
                "query": "excitement breakthrough discovery",
                "search_type": "emotional",
                "limit": 10,
            },
        )
        data = json.loads(result.content[0].text)

        # search_shortterm returns data directly, no "success" field
        assert "memories" in data

        # Should return results (even if none match perfectly)
        assert isinstance(data["memories"], list)

        # If we found results, they should have emotional similarity scores
        for memory in data["memories"]:
            assert "similarity_score" in memory
            assert 0.0 <= memory["similarity_score"] <= 1.0

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_shortterm"
        assert (
            latest_duration < 300
        ), f"search_shortterm took {latest_duration:.1f}ms, should be <300ms"

        print(f"💭 STM emotional search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_shortterm_with_temporal_filtering(test_stack_seeded):
    """Test STM search with temporal range filtering."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search within a specific time window
        result = await time_mcp_call(
            client,
            "search_shortterm",
            {
                "query": "memory system",
                "search_type": "semantic",
                "limit": 15,
                "through_the_last": "6h",
            },
        )
        data = json.loads(result.content[0].text)

        # search_shortterm returns data directly, no "success" field
        assert "memories" in data

        # Should return results within time window
        assert isinstance(data["memories"], list)

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_shortterm"
        assert (
            latest_duration < 300
        ), f"search_shortterm took {latest_duration:.1f}ms, should be <300ms"

        print(f"🕰️ Temporal STM search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_shortterm_memory_cross_system_consistency(test_stack_seeded):
    """Test that STM data appears correctly in unified search."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Store a distinctive STM memory
        store_result = await time_mcp_call(
            client,
            "remember_shortterm",
            {
                "content": "Cross-system test marker: unique STM content for unified search validation"
            },
        )
        store_data = json.loads(store_result.content[0].text)
        assert store_data["status"] == "stored"

        # Search for it in unified search
        unified_result = await time_mcp_call(
            client,
            "search_all_memories",
            {"query": "Cross-system test marker unique STM content", "limit": 10},
        )
        unified_data = json.loads(unified_result.content[0].text)

        assert unified_data["success"] is True
        assert len(unified_data["results"]) > 0

        # Should find our STM memory in unified results
        stm_sources = [
            r
            for r in unified_data["results"]
            if r["source"] in ["STM", "STM_SEMANTIC", "STM_EMOTIONAL"]
        ]
        assert len(stm_sources) > 0, "Should find STM results in unified search"

        # Verify content matches
        found_content = False
        for result in unified_data["results"]:
            if "Cross-system test marker" in result["content"]:
                found_content = True
                break
        assert found_content, "Should find our test content in unified search"

        # Assert performance for both operations
        from tests.e2e.fixtures.performance import collector

        # Get timing for both calls
        store_duration = None
        unified_duration = None
        for metric in reversed(collector.get_metrics()):
            if (
                metric["operation"] == "mcp_call_remember_shortterm"
                and store_duration is None
            ):
                store_duration = metric["duration_ms"]
            elif (
                metric["operation"] == "mcp_call_search_all_memories"
                and unified_duration is None
            ):
                unified_duration = metric["duration_ms"]
            if store_duration is not None and unified_duration is not None:
                break

        assert (
            store_duration is not None
        ), "Should have recorded timing for remember_shortterm"
        assert (
            unified_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            store_duration < 300
        ), f"remember_shortterm took {store_duration:.1f}ms, should be <300ms"
        assert (
            unified_duration < 500
        ), f"search_all_memories took {unified_duration:.1f}ms, should be <500ms"

        print(
            f"🔄 Cross-system: store {store_duration:.1f}ms, search {unified_duration:.1f}ms"
        )


@pytest.mark.asyncio
@performance_test
async def test_shortterm_memory_performance_comprehensive(test_stack_seeded):
    """Comprehensive STM performance test across all operations."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Test remember_shortterm performance
        remember_result = await time_mcp_call(
            client,
            "remember_shortterm",
            {
                "content": "Performance test memory with rich semantic and emotional content for comprehensive evaluation"
            },
        )

        # Test browse_shortterm performance
        browse_result = await time_mcp_call(
            client, "browse_shortterm", {"limit": 20, "order": "desc"}
        )

        # Test semantic search performance
        semantic_result = await time_mcp_call(
            client,
            "search_shortterm",
            {
                "query": "performance evaluation comprehensive",
                "search_type": "semantic",
                "limit": 10,
            },
        )

        # Test emotional search performance
        emotional_result = await time_mcp_call(
            client,
            "search_shortterm",
            {
                "query": "evaluation comprehensive",
                "search_type": "emotional",
                "limit": 10,
            },
        )

        # Verify all operations succeeded
        assert json.loads(remember_result.content[0].text)["status"] == "stored"
        # browse_shortterm and search_shortterm return data directly, no "success" field
        assert "memories" in json.loads(browse_result.content[0].text)
        assert "memories" in json.loads(semantic_result.content[0].text)
        assert "memories" in json.loads(emotional_result.content[0].text)

        # Assert performance expectations for all operations
        from tests.e2e.fixtures.performance import collector

        metrics = collector.get_metrics()
        recent_metrics = metrics[-4:]  # Get the 4 most recent metrics

        for metric in recent_metrics:
            operation = metric["operation"]
            duration = metric["duration_ms"]

            if operation == "mcp_call_remember_shortterm":
                assert (
                    duration < 300
                ), f"remember_shortterm took {duration:.1f}ms, should be <300ms"
            elif operation == "mcp_call_browse_shortterm":
                assert (
                    duration < 100
                ), f"browse_shortterm took {duration:.1f}ms, should be <100ms"
            elif operation == "mcp_call_search_shortterm":
                assert (
                    duration < 300
                ), f"search_shortterm took {duration:.1f}ms, should be <300ms"

        print(
            "🎯 STM comprehensive performance test completed - all operations within thresholds"
        )

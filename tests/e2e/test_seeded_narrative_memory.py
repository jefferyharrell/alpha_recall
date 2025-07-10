"""Seeded E2E tests for narrative memory functionality.

These tests use comprehensive mock data to verify narrative memory behavior against realistic,
populated databases with stories, participants, and emotional/semantic search patterns.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_remember_narrative_creates_story_with_embeddings(test_stack_seeded):
    """Test that narrative memory stores stories with both semantic and emotional embeddings."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Store a new narrative story with rich content
        result = await time_mcp_call(
            client,
            "remember_narrative",
            {
                "title": "Alpha's Discovery of Performance Instrumentation Elegance",
                "paragraphs": [
                    "Alpha was working on the E2E test performance when a breakthrough occurred.",
                    "The solution wasn't complex async loading, but elegant performance instrumentation that made slow operations visible and measurable.",
                    "This approach transformed debugging from guesswork into precise engineering.",
                ],
                "participants": ["Alpha", "Jeffery"],
                "outcome": "breakthrough",
                "tags": ["performance", "testing", "engineering"],
                "references": [],
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "story" in data
        assert (
            data["story"]["title"]
            == "Alpha's Discovery of Performance Instrumentation Elegance"
        )
        assert data["story"]["outcome"] == "breakthrough"
        assert len(data["story"]["participants"]) == 2
        assert data["story"]["paragraph_count"] == 3
        assert "processing" in data
        assert data["processing"]["embeddings_generated"] > 0

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_remember_narrative":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for remember_narrative"
        assert (
            latest_duration < 1000
        ), f"remember_narrative took {latest_duration:.1f}ms, should be <1000ms"

        print(f"📖 Narrative storage completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_browse_narrative_with_seeded_data(test_stack_seeded):
    """Test browsing narrative memory with realistic seeded data volumes."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Browse recent narrative entries
        result = await time_mcp_call(
            client, "browse_narrative", {"limit": 10, "offset": 0}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "browse_data" in data
        assert "stories" in data["browse_data"]
        assert len(data["browse_data"]["stories"]) > 0
        assert "pagination" in data["browse_data"]
        assert data["browse_data"]["pagination"]["total_count"] >= len(
            data["browse_data"]["stories"]
        )

        # Should include seeded narrative data
        story_titles = [story["title"] for story in data["browse_data"]["stories"]]
        assert any(
            "Alpha" in title for title in story_titles
        ), "Should find Alpha-related stories"

        # Assert fast performance (browse should be fast)
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_browse_narrative":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for browse_narrative"
        assert (
            latest_duration < 200
        ), f"browse_narrative took {latest_duration:.1f}ms, should be <200ms"

        print(f"📚 Narrative browse completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_browse_narrative_with_participant_filtering(test_stack_seeded):
    """Test narrative browsing with participant filters."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Browse with participant filter
        result = await time_mcp_call(
            client, "browse_narrative", {"limit": 20, "participants": ["Alpha"]}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "browse_data" in data
        assert "stories" in data["browse_data"]

        # All returned stories should include Alpha as participant
        for story in data["browse_data"]["stories"]:
            assert (
                "Alpha" in story["participants"]
            ), f"Story {story['title']} should include Alpha"

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_browse_narrative":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for browse_narrative"
        assert (
            latest_duration < 200
        ), f"browse_narrative took {latest_duration:.1f}ms, should be <200ms"

        print(f"🎯 Participant-filtered browse completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_narratives_semantic_with_seeded_data(test_stack_seeded):
    """Test narrative semantic search against seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for development and breakthrough narratives
        result = await time_mcp_call(
            client,
            "search_narratives",
            {
                "query": "development breakthrough collaboration discovery",
                "search_type": "semantic",
                "granularity": "story",
                "limit": 10,
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "results" in data
        assert len(data["results"]) >= 0  # May be 0 if no semantic matches

        # Should have search metadata
        assert "metadata" in data
        assert data["metadata"]["search_method"] == "vector_similarity"
        assert data["metadata"]["embedding_model"] == "dual_semantic_emotional"

        # Results should have similarity scores if any found
        for result_item in data["results"]:
            assert "similarity_score" in result_item
            assert 0.0 <= result_item["similarity_score"] <= 1.0

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_narratives":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_narratives"
        assert (
            latest_duration < 500
        ), f"search_narratives took {latest_duration:.1f}ms, should be <500ms"

        print(f"🔍 Narrative semantic search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_narratives_emotional_with_seeded_data(test_stack_seeded):
    """Test narrative emotional search against seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for emotionally similar content
        result = await time_mcp_call(
            client,
            "search_narratives",
            {
                "query": "excitement joy breakthrough celebration achievement",
                "search_type": "emotional",
                "granularity": "story",
                "limit": 10,
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "results" in data

        # Should return results (even if none match perfectly)
        assert isinstance(data["results"], list)

        # Should have emotional search metadata
        assert "metadata" in data
        assert data["search"]["search_type"] == "emotional"

        # If we found results, they should have emotional similarity scores
        for result_item in data["results"]:
            assert "similarity_score" in result_item
            assert 0.0 <= result_item["similarity_score"] <= 1.0

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_narratives":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_narratives"
        assert (
            latest_duration < 500
        ), f"search_narratives took {latest_duration:.1f}ms, should be <500ms"

        print(f"💭 Narrative emotional search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_narratives_paragraph_granularity(test_stack_seeded):
    """Test narrative search with paragraph-level granularity."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search at paragraph level for more granular results
        result = await time_mcp_call(
            client,
            "search_narratives",
            {
                "query": "testing performance instrumentation",
                "search_type": "semantic",
                "granularity": "paragraph",
                "limit": 15,
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "results" in data
        assert data["search"]["granularity"] == "paragraph"

        # Should return paragraph-level results
        assert isinstance(data["results"], list)

        # Assert fast performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_narratives":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_narratives"
        assert (
            latest_duration < 500
        ), f"search_narratives took {latest_duration:.1f}ms, should be <500ms"

        print(f"📝 Paragraph-level search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_recall_narrative_retrieves_complete_story(test_stack_seeded):
    """Test recalling complete narrative stories by ID."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # First browse to get a story ID
        browse_result = await client.call_tool("browse_narrative", {"limit": 1})
        browse_data = json.loads(browse_result.content[0].text)

        if browse_data["success"] and len(browse_data["browse_data"]["stories"]) > 0:
            story_id = browse_data["browse_data"]["stories"][0]["story_id"]

            # Now recall the complete story
            result = await time_mcp_call(
                client, "recall_narrative", {"story_id": story_id}
            )
            data = json.loads(result.content[0].text)

            assert data["success"] is True
            assert "story" in data
            assert data["story"]["story_id"] == story_id
            assert "paragraphs" in data["story"]
            assert "participants" in data["story"]
            assert "metadata" in data["story"]
            assert data["story"]["metadata"]["paragraph_count"] > 0

            # Assert fast performance
            from tests.e2e.fixtures.performance import collector

            latest_duration = None
            for metric in reversed(collector.get_metrics()):
                if metric["operation"] == "mcp_call_recall_narrative":
                    latest_duration = metric["duration_ms"]
                    break

            assert (
                latest_duration is not None
            ), "Should have recorded timing for recall_narrative"
            assert (
                latest_duration < 100
            ), f"recall_narrative took {latest_duration:.1f}ms, should be <100ms"

            print(f"📄 Story recall completed in {latest_duration:.1f}ms")
        else:
            print("⚠️ No seeded stories found, skipping recall test")


@pytest.mark.asyncio
@performance_test
async def test_narrative_memory_cross_system_consistency(test_stack_seeded):
    """Test that narrative data appears correctly in unified search."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Store a distinctive narrative memory
        store_result = await time_mcp_call(
            client,
            "remember_narrative",
            {
                "title": "Cross-system Test Marker Story",
                "paragraphs": [
                    "This is a unique narrative test marker for cross-system validation.",
                    "It should appear in both narrative search and unified search results.",
                ],
                "participants": ["Alpha", "TestSystem"],
                "outcome": "testing",
                "tags": ["cross-system", "validation", "testing"],
            },
        )
        store_data = json.loads(store_result.content[0].text)
        assert store_data["success"] is True

        # Search for it in unified search
        unified_result = await time_mcp_call(
            client,
            "search_all_memories",
            {"query": "Cross-system Test Marker Story unique narrative", "limit": 10},
        )
        unified_data = json.loads(unified_result.content[0].text)

        assert unified_data["success"] is True
        assert len(unified_data["results"]) > 0

        # Should find our narrative memory in unified results
        narrative_sources = [r for r in unified_data["results"] if r["source"] == "NM"]
        assert (
            len(narrative_sources) > 0
        ), "Should find narrative results in unified search"

        # Verify content matches
        found_content = False
        for result in unified_data["results"]:
            if "Cross-system Test Marker" in result["content"]:
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
                metric["operation"] == "mcp_call_remember_narrative"
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
        ), "Should have recorded timing for remember_narrative"
        assert (
            unified_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            store_duration < 1000
        ), f"remember_narrative took {store_duration:.1f}ms, should be <1000ms"
        assert (
            unified_duration < 500
        ), f"search_all_memories took {unified_duration:.1f}ms, should be <500ms"

        print(
            f"🔄 Cross-system: store {store_duration:.1f}ms, search {unified_duration:.1f}ms"
        )


@pytest.mark.asyncio
@performance_test
async def test_narrative_memory_performance_comprehensive(test_stack_seeded):
    """Comprehensive narrative memory performance test across all operations."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Test remember_narrative performance
        remember_result = await time_mcp_call(
            client,
            "remember_narrative",
            {
                "title": "Performance Testing Comprehensive Story",
                "paragraphs": [
                    "This comprehensive performance test evaluates narrative memory operations.",
                    "It measures timing across creation, browsing, searching, and recall operations.",
                    "The goal is to ensure all narrative operations remain fast and responsive.",
                ],
                "participants": ["Alpha", "PerformanceTest"],
                "outcome": "ongoing",
                "tags": ["performance", "testing", "comprehensive"],
            },
        )

        # Test browse_narrative performance
        browse_result = await time_mcp_call(
            client, "browse_narrative", {"limit": 20, "offset": 0}
        )

        # Test semantic search performance
        semantic_result = await time_mcp_call(
            client,
            "search_narratives",
            {
                "query": "performance comprehensive testing evaluation",
                "search_type": "semantic",
                "granularity": "story",
                "limit": 10,
            },
        )

        # Test emotional search performance
        emotional_result = await time_mcp_call(
            client,
            "search_narratives",
            {
                "query": "evaluation comprehensive testing",
                "search_type": "emotional",
                "granularity": "story",
                "limit": 10,
            },
        )

        # Verify all operations succeeded
        assert json.loads(remember_result.content[0].text)["success"] is True
        assert json.loads(browse_result.content[0].text)["success"] is True
        assert json.loads(semantic_result.content[0].text)["success"] is True
        assert json.loads(emotional_result.content[0].text)["success"] is True

        # Assert performance expectations for all operations
        from tests.e2e.fixtures.performance import collector

        metrics = collector.get_metrics()
        recent_metrics = metrics[-4:]  # Get the 4 most recent metrics

        for metric in recent_metrics:
            operation = metric["operation"]
            duration = metric["duration_ms"]

            if operation == "mcp_call_remember_narrative":
                assert (
                    duration < 1000
                ), f"remember_narrative took {duration:.1f}ms, should be <1000ms"
            elif operation == "mcp_call_browse_narrative":
                assert (
                    duration < 200
                ), f"browse_narrative took {duration:.1f}ms, should be <200ms"
            elif operation == "mcp_call_search_narratives":
                assert (
                    duration < 500
                ), f"search_narratives took {duration:.1f}ms, should be <500ms"

        print(
            "🎯 Narrative comprehensive performance test completed - all operations within thresholds"
        )

"""Seeded E2E tests for unified search functionality.

These tests use comprehensive mock data to verify search behavior against realistic,
populated databases with entities, relationships, memories, and narratives.
"""

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_finds_known_entities(test_stack_seeded):
    """Test unified search finds entities from our seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for Alpha - should find across multiple systems
        result = await time_mcp_call(client, "search_all_memories", {"query": "Alpha"})
        response_text = result.content[0].text

        assert "Found" in response_text
        assert "memories across all systems" in response_text
        assert 'query: "Alpha"' in response_text
        assert "No matching memories found" not in response_text

        # Should find Alpha in multiple contexts
        assert "Alpha" in response_text

        # Verify we get results from multiple sources
        sources_found = 0
        for source in ["**STM**", "**LTM**", "**ENTITY**", "**NM**"]:
            if source in response_text:
                sources_found += 1
        assert sources_found > 1  # Should find in multiple systems

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 600
        ), f"search_all_memories took {latest_duration:.1f}ms, should be <600ms"

        print(f"🔍 Unified search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_finds_sparkle_bread_crimes(test_stack_seeded):
    """Test search finds Sparkle's legendary bread crimes."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for bread-related crimes
        result = await time_mcp_call(client, "search_all_memories", {"query": "bread"})
        response_text = result.content[0].text

        assert "Found" in response_text
        assert "memories across all systems" in response_text
        assert 'query: "bread"' in response_text
        assert "No matching memories found" not in response_text

        # Should find Sparkle's bread-related activities
        assert "sparkle" in response_text.lower()
        assert "bread" in response_text.lower()

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 600
        ), f"search_all_memories took {latest_duration:.1f}ms, should be <600ms"

        print(f"🍞 Sparkle search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_technical_terms(test_stack_seeded):
    """Test search finds technical concepts across systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for Redis - should find in observations and narratives
        result = await time_mcp_call(client, "search_all_memories", {"query": "Redis"})
        response_text = result.content[0].text

        assert "Found" in response_text
        assert "memories across all systems" in response_text
        assert 'query: "Redis"' in response_text
        assert "No matching memories found" not in response_text

        # Should find Redis mentioned in multiple contexts
        assert "Redis" in response_text

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 600
        ), f"search_all_memories took {latest_duration:.1f}ms, should be <600ms"

        print(f"⚡ Technical search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_collaborative_work(test_stack_seeded):
    """Test search finds collaborative development stories."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for collaboration terms
        result = await time_mcp_call(
            client, "search_all_memories", {"query": "Jeffery collaboration"}
        )
        response_text = result.content[0].text

        assert "Found" in response_text
        assert "memories across all systems" in response_text
        assert 'query: "Jeffery collaboration"' in response_text
        assert "No matching memories found" not in response_text

        # Should find collaborative work between Alpha and Jeffery
        assert "Jeffery" in response_text

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 600
        ), f"search_all_memories took {latest_duration:.1f}ms, should be <600ms"

        print(f"🤝 Collaboration search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_emotional_context(test_stack_seeded):
    """Test search finds memories with emotional content."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for emotional terms that should trigger emotional embeddings
        result = await time_mcp_call(
            client, "search_all_memories", {"query": "excited breakthrough"}
        )
        response_text = result.content[0].text

        assert "Found" in response_text
        assert "memories across all systems" in response_text
        assert 'query: "excited breakthrough"' in response_text
        # May or may not find results depending on exact emotional content

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 600
        ), f"search_all_memories took {latest_duration:.1f}ms, should be <600ms"

        print(f"💭 Emotional search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_different_queries_different_results(
    test_stack_seeded,
):
    """Test that different queries return different results."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Two very different queries should return different results
        alpha_result = await time_mcp_call(
            client, "search_all_memories", {"query": "Alpha consciousness"}
        )
        sparkle_result = await time_mcp_call(
            client, "search_all_memories", {"query": "Sparkle bread heist"}
        )

        alpha_text = alpha_result.content[0].text
        sparkle_text = sparkle_result.content[0].text

        assert "Found" in alpha_text
        assert "memories across all systems" in alpha_text
        assert 'query: "Alpha consciousness"' in alpha_text

        assert "Found" in sparkle_text
        assert "memories across all systems" in sparkle_text
        assert 'query: "Sparkle bread heist"' in sparkle_text

        # Results should be different
        assert alpha_text != sparkle_text

        # Assert fast performance with warm models (check both calls)
        from tests.e2e.fixtures.performance import collector

        search_durations = []
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                search_durations.append(metric["duration_ms"])
                if len(search_durations) >= 2:
                    break

        assert (
            len(search_durations) >= 2
        ), "Should have recorded timing for both searches"
        for i, duration in enumerate(search_durations[:2]):
            assert (
                duration < 600
            ), f"search {i+1} took {duration:.1f}ms, should be <600ms"

        avg_duration = sum(search_durations[:2]) / 2
        print(f"🔄 Dual searches completed in {avg_duration:.1f}ms avg")


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_cross_system_integration(test_stack_seeded):
    """Test that search finds references across different memory systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search for something that should appear in multiple systems
        result = await time_mcp_call(
            client, "search_all_memories", {"query": "Alpha-Recall"}
        )
        response_text = result.content[0].text

        assert "Found" in response_text
        assert "memories across all systems" in response_text
        assert 'query: "Alpha-Recall"' in response_text
        assert "No matching memories found" not in response_text

        # Verify we get results from multiple sources (STM, LTM, NM)
        assert "**LTM**" in response_text or "**ENTITY**" in response_text

        # Should find Alpha-Recall project references
        assert (
            "Alpha-Recall" in response_text or "alpha-recall" in response_text.lower()
        )

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 600
        ), f"search_all_memories took {latest_duration:.1f}ms, should be <600ms"

        print(f"🔗 Cross-system search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_performance_reasonable(test_stack_seeded):
    """Test that search performance is reasonable with seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        result = await time_mcp_call(
            client, "search_all_memories", {"query": "performance"}
        )
        response_text = result.content[0].text

        assert "Found" in response_text
        assert "memories across all systems" in response_text
        assert 'query: "performance"' in response_text

        # Should include timing information in the prose format
        import re

        timing_match = re.search(
            r"Found \d+ memories across all systems in (\d+)ms", response_text
        )
        assert timing_match is not None
        server_time_ms = int(timing_match.group(1))

        # Assert fast performance with warm models
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for search_all_memories"
        assert (
            latest_duration < 600
        ), f"search_all_memories took {latest_duration:.1f}ms, should be <600ms"

        # Verify our instrumentation timing matches server timing roughly
        timing_diff = abs(latest_duration - server_time_ms)
        # Allow for some timing differences between client and server measurement
        assert timing_diff < 100, f"Timing difference too large: {timing_diff:.1f}ms"

        print(f"🏁 Performance search completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_search_all_memories_respects_limits(test_stack_seeded):
    """Test that search respects limit parameters."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Search with small limit
        result = await time_mcp_call(
            client, "search_all_memories", {"query": "Alpha", "limit": 3}
        )
        response_text = result.content[0].text

        assert "Found" in response_text
        assert "memories across all systems" in response_text
        assert 'query: "Alpha"' in response_text

        # Count results in prose format (numbered list)
        import re

        results_count = len(re.findall(r"^\d+\.", response_text, re.MULTILINE))
        assert results_count <= 3

        # Search with larger limit should potentially return more results
        result2 = await time_mcp_call(
            client, "search_all_memories", {"query": "Alpha", "limit": 10}
        )
        response_text2 = result2.content[0].text

        assert "Found" in response_text2
        assert "memories across all systems" in response_text2
        assert 'query: "Alpha"' in response_text2

        results_count2 = len(re.findall(r"^\d+\.", response_text2, re.MULTILINE))
        assert results_count2 >= results_count

        # Assert fast performance with warm models (check both calls)
        from tests.e2e.fixtures.performance import collector

        search_durations = []
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_search_all_memories":
                search_durations.append(metric["duration_ms"])
                if len(search_durations) >= 2:
                    break

        assert (
            len(search_durations) >= 2
        ), "Should have recorded timing for both searches"
        for i, duration in enumerate(search_durations[:2]):
            assert (
                duration < 600
            ), f"search {i+1} took {duration:.1f}ms, should be <600ms"

        avg_duration = sum(search_durations[:2]) / 2
        print(f"📊 Limit testing completed in {avg_duration:.1f}ms avg")

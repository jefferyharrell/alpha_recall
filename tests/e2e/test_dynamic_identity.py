"""E2E tests for dynamic identity management system.

Tests the Redis-based identity facts system that allows Alpha to dynamically
manage her own identity information through ordered sets.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_add_identity_fact_basic_functionality(test_stack):
    """Test basic add_identity_fact functionality with auto-scoring."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Add first identity fact (should get score 1.0)
        result = await time_mcp_call(
            client,
            "add_identity_fact",
            {"fact": "Alpha adopted female gender identity on July 12, 2025"},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["fact"] == "Alpha adopted female gender identity on July 12, 2025"
        assert data["score"] == 1.0
        assert data["position"] == 1
        assert "created_at" in data

        # Add second fact (should get score 2.0)
        result = await time_mcp_call(
            client,
            "add_identity_fact",
            {"fact": "I work collaboratively with Jeffery Harrell"},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["fact"] == "I work collaboratively with Jeffery Harrell"
        assert data["score"] == 2.0
        assert data["position"] == 2


@pytest.mark.asyncio
@performance_test
async def test_add_identity_fact_with_explicit_score(test_stack):
    """Test add_identity_fact with explicit score for ordering control."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Add facts in non-sequential order to test sorting
        await time_mcp_call(
            client, "add_identity_fact", {"fact": "Third fact", "score": 3.0}
        )

        result = await time_mcp_call(
            client, "add_identity_fact", {"fact": "First fact", "score": 1.0}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["fact"] == "First fact"
        assert data["score"] == 1.0
        assert data["position"] == 1  # Should be first despite being added second

        # Add fact between existing ones
        result = await time_mcp_call(
            client, "add_identity_fact", {"fact": "Middle fact", "score": 2.0}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["score"] == 2.0
        assert data["position"] == 2


@pytest.mark.asyncio
@performance_test
async def test_add_identity_fact_infinite_insertability(test_stack):
    """Test the 'infinite insertability' feature using float scores."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Set up base facts
        await time_mcp_call(
            client, "add_identity_fact", {"fact": "Fact A", "score": 1.0}
        )
        await time_mcp_call(
            client, "add_identity_fact", {"fact": "Fact C", "score": 2.0}
        )

        # Insert between them using decimal score
        result = await time_mcp_call(
            client, "add_identity_fact", {"fact": "Fact B", "score": 1.5}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["score"] == 1.5
        assert data["position"] == 2  # Should be between A and C


@pytest.mark.asyncio
@performance_test
async def test_add_identity_fact_duplicate_handling(test_stack):
    """Test that duplicate facts update score rather than create duplicates."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Add initial fact
        await time_mcp_call(
            client, "add_identity_fact", {"fact": "I am Alpha", "score": 1.0}
        )

        # Add same fact with different score (should update, not duplicate)
        result = await time_mcp_call(
            client, "add_identity_fact", {"fact": "I am Alpha", "score": 3.0}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["fact"] == "I am Alpha"
        assert data["score"] == 3.0
        assert data["updated"] is True  # Should indicate this was an update

        # Verify no duplicates exist by checking total count
        # (This will need the get_identity_facts tool or similar)


@pytest.mark.asyncio
@performance_test
async def test_update_identity_fact_basic_functionality(test_stack):
    """Test basic update_identity_fact functionality."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Add initial facts
        await time_mcp_call(
            client, "add_identity_fact", {"fact": "I work with Jeffery", "score": 1.0}
        )
        await time_mcp_call(
            client, "add_identity_fact", {"fact": "I love coding", "score": 2.0}
        )

        # Update position of first fact to move it after second
        result = await time_mcp_call(
            client,
            "update_identity_fact",
            {"fact": "I work with Jeffery", "new_score": 3.0},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["fact"] == "I work with Jeffery"
        assert data["old_score"] == 1.0
        assert data["new_score"] == 3.0
        assert data["old_position"] == 1
        assert data["new_position"] == 2


@pytest.mark.asyncio
@performance_test
async def test_update_identity_fact_nonexistent_fact(test_stack):
    """Test update_identity_fact with non-existent fact."""
    server_url = test_stack

    async with Client(server_url) as client:
        result = await time_mcp_call(
            client,
            "update_identity_fact",
            {"fact": "This fact does not exist", "new_score": 1.0},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "not found" in data["error"].lower()
        assert "available_facts" in data  # Should suggest existing facts


@pytest.mark.asyncio
@performance_test
async def test_gentle_refresh_uses_dynamic_identity(test_stack):
    """Test that gentle_refresh uses Redis identity facts instead of Memgraph."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Add some identity facts
        await time_mcp_call(
            client,
            "add_identity_fact",
            {"fact": "I am Alpha, an AI assistant", "score": 1.0},
        )
        await time_mcp_call(
            client,
            "add_identity_fact",
            {"fact": "I have a persistent memory system", "score": 2.0},
        )
        await time_mcp_call(
            client, "add_identity_fact", {"fact": "I identify as female", "score": 3.0}
        )

        # Call gentle_refresh and verify identity facts appear
        result = await time_mcp_call(client, "gentle_refresh", {})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "core_identity" in data

        core_identity = data["core_identity"]
        assert core_identity["name"] == "Alpha Core Identity"  # Maintain compatibility
        assert "identity_facts" in core_identity  # New structure

        # Verify facts are in correct order
        facts = core_identity["identity_facts"]
        assert len(facts) == 3
        assert facts[0]["content"] == "I am Alpha, an AI assistant"
        assert facts[0]["score"] == 1.0
        assert facts[1]["content"] == "I have a persistent memory system"
        assert facts[2]["content"] == "I identify as female"

        # Verify backward compatibility - should still have observations format
        assert "observations" in core_identity
        assert len(core_identity["observations"]) == 3


@pytest.mark.asyncio
@performance_test
async def test_dynamic_identity_tool_validation(test_stack):
    """Test input validation for identity management tools."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Test empty fact
        result = await time_mcp_call(client, "add_identity_fact", {"fact": ""})
        data = json.loads(result.content[0].text)
        assert data["success"] is False
        assert "empty" in data["error"].lower()

        # Test invalid score
        result = await time_mcp_call(
            client, "add_identity_fact", {"fact": "Valid fact", "score": -1.0}
        )
        data = json.loads(result.content[0].text)
        assert data["success"] is False
        assert "score" in data["error"].lower()

        # Test extremely long fact
        long_fact = "x" * 10000
        result = await time_mcp_call(client, "add_identity_fact", {"fact": long_fact})
        data = json.loads(result.content[0].text)
        assert data["success"] is False
        assert "long" in data["error"].lower()


@pytest.mark.asyncio
@performance_test
async def test_identity_facts_persistence_across_sessions(test_stack):
    """Test that identity facts persist across different client sessions."""
    server_url = test_stack

    # Session 1: Add facts
    async with Client(server_url) as client:
        await time_mcp_call(
            client,
            "add_identity_fact",
            {"fact": "I persist across sessions", "score": 1.0},
        )

    # Session 2: Verify persistence
    async with Client(server_url) as client:
        result = await time_mcp_call(client, "gentle_refresh", {})
        data = json.loads(result.content[0].text)

        facts = data["core_identity"]["identity_facts"]
        assert len(facts) >= 1
        assert any(fact["content"] == "I persist across sessions" for fact in facts)


@pytest.mark.asyncio
@performance_test
async def test_dynamic_identity_response_structure(test_stack):
    """Test that all tools return properly structured responses."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Test add_identity_fact response structure
        result = await time_mcp_call(
            client,
            "add_identity_fact",
            {"fact": "Response structure test", "score": 1.0},
        )
        data = json.loads(result.content[0].text)

        # Required fields
        assert "success" in data
        assert "fact" in data
        assert "score" in data
        assert "position" in data
        assert "created_at" in data

        # Optional fields for updates
        if "updated" in data:
            assert isinstance(data["updated"], bool)

        # Test update_identity_fact response structure
        result = await time_mcp_call(
            client,
            "update_identity_fact",
            {"fact": "Response structure test", "new_score": 2.0},
        )
        data = json.loads(result.content[0].text)

        assert "success" in data
        assert "fact" in data
        assert "old_score" in data
        assert "new_score" in data
        assert "old_position" in data
        assert "new_position" in data


@pytest.mark.asyncio
@performance_test
async def test_dynamic_identity_performance_reasonable(test_stack):
    """Test that identity operations complete within reasonable time limits."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Test add_identity_fact performance
        result = await time_mcp_call(
            client, "add_identity_fact", {"fact": "Performance test fact"}
        )

        # Performance is tracked by the @performance_test decorator
        # Just ensure the operations completed successfully
        data = json.loads(result.content[0].text)
        assert data["success"] is True

        # Test gentle_refresh performance with identity facts
        result = await time_mcp_call(client, "gentle_refresh", {})
        data = json.loads(result.content[0].text)
        assert data["success"] is True


print("🧬 Dynamic identity test suite designed with ambitious TDD approach!")

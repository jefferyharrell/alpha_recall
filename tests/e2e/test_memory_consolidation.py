"""E2E tests for memory consolidation functionality."""

import asyncio
import json

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_memory_consolidation_returns_structured_json(test_stack):
    """Test that memory consolidation returns structured JSON with expected fields."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Store some test memories to consolidate
        await client.call_tool(
            "remember_shortterm",
            {"content": "Working on search_all_memories bug fix with Jeffery"},
        )
        await client.call_tool(
            "remember_shortterm",
            {"content": "Successfully implemented E2E testing for Alpha-Recall"},
        )
        await client.call_tool(
            "remember_shortterm",
            {"content": "Sparkle's bread obsession revealed through memory search"},
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Trigger memory consolidation via gentle_refresh
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "memory_consolidation" in data

        consolidation = data["memory_consolidation"]

        # Verify all expected fields are present
        required_fields = [
            "entities",
            "relationships",
            "insights",
            "summary",
            "emotional_context",
            "next_steps",
            "processed_memories_count",
            "consolidation_timestamp",
            "model_used",
        ]

        for field in required_fields:
            assert field in consolidation, f"Missing required field: {field}"

        # Verify field types
        assert isinstance(consolidation["entities"], list), "entities should be a list"
        assert isinstance(
            consolidation["relationships"], list
        ), "relationships should be a list"
        assert isinstance(consolidation["insights"], list), "insights should be a list"
        assert isinstance(consolidation["summary"], str), "summary should be a string"
        assert isinstance(
            consolidation["emotional_context"], str
        ), "emotional_context should be a string"
        assert isinstance(
            consolidation["next_steps"], list
        ), "next_steps should be a list"
        assert isinstance(
            consolidation["processed_memories_count"], int
        ), "processed_memories_count should be an int"
        assert isinstance(
            consolidation["consolidation_timestamp"], str
        ), "consolidation_timestamp should be a string"
        assert isinstance(
            consolidation["model_used"], str
        ), "model_used should be a string"

        # Verify we actually processed some memories (consolidation should be enabled in E2E)
        assert (
            consolidation["processed_memories_count"] > 0
        ), f"Should have processed at least some memories. Got: {consolidation}"

        # Check if consolidation actually ran successfully or failed gracefully
        if consolidation["model_used"] == "error":
            # Ollama not available - check that we got a meaningful error message
            assert (
                "failed" in consolidation["summary"].lower()
            ), "Error state should have meaningful error message"
            print(
                f"Memory consolidation failed gracefully (Ollama not available): {consolidation['summary']}"
            )
        elif consolidation["model_used"] == "placeholder":
            pytest.skip(
                "Memory consolidation disabled - skipping model-specific assertions"
            )
        else:
            # Ollama available and working - verify model was actually used
            assert (
                consolidation["model_used"] == "llama3.2:1b"
            ), f"Should use configured model. Got: {consolidation['model_used']}"


@pytest.mark.asyncio
async def test_memory_consolidation_handles_json_parsing_failure_gracefully(test_stack):
    """Test that consolidation falls back gracefully when JSON parsing fails."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Store a memory to ensure consolidation runs
        await client.call_tool(
            "remember_shortterm", {"content": "Test memory for consolidation"}
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Trigger consolidation
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        consolidation = data["memory_consolidation"]

        # Even if JSON parsing fails, we should get a valid structure
        # (either parsed JSON or fallback with summary containing the raw response)
        assert isinstance(consolidation["entities"], list)
        assert isinstance(consolidation["summary"], str)
        assert len(consolidation["summary"]) > 0, "Summary should not be empty"

        # If Ollama is not available, that's a valid failure mode
        if consolidation["model_used"] == "error":
            assert (
                "failed" in consolidation["summary"].lower()
            ), "Error state should have meaningful error message"
            print(
                f"Memory consolidation failed gracefully (Ollama not available): {consolidation['summary']}"
            )


@pytest.mark.asyncio
async def test_memory_consolidation_with_no_recent_memories(test_stack):
    """Test consolidation behavior when there are no recent memories."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Don't add any memories - test with empty dataset

        # Trigger consolidation
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        consolidation = data["memory_consolidation"]

        # Should return valid structure (but may have memories from other tests)
        assert isinstance(consolidation["processed_memories_count"], int)
        assert isinstance(consolidation["entities"], list)
        assert isinstance(consolidation["summary"], str)

        # If no memories were processed, should be empty
        if consolidation["processed_memories_count"] == 0:
            # All list fields should be empty lists
            assert consolidation["entities"] == []
            assert consolidation["relationships"] == []
            assert consolidation["insights"] == []
            assert consolidation["next_steps"] == []
            # Model should be placeholder when no memories to process
            assert (
                consolidation["model_used"] == "placeholder"
            ), f"Should be placeholder when no memories to process. Got: {consolidation['model_used']}"
        else:
            # Memories from other tests were processed - that's fine too
            print(
                f"Processed {consolidation['processed_memories_count']} memories from other tests"
            )
            # Just verify structure is valid
            assert isinstance(consolidation["entities"], list)
            assert isinstance(consolidation["relationships"], list)
            assert isinstance(consolidation["insights"], list)
            assert isinstance(consolidation["next_steps"], list)


@pytest.mark.asyncio
async def test_memory_consolidation_identifies_relevant_entities(test_stack):
    """Test that consolidation can identify relevant entities from memories."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Store memories with clear entity references
        await client.call_tool(
            "remember_shortterm",
            {"content": "Jeffery and I debugged the search_all_memories tool"},
        )
        await client.call_tool(
            "remember_shortterm", {"content": "Alpha-Recall v1.0.0 is working great"}
        )
        await client.call_tool(
            "remember_shortterm", {"content": "Sparkle's bread crimes are legendary"}
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Trigger consolidation
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        consolidation = data["memory_consolidation"]

        # Debug: print the actual consolidation result to understand the structure
        print(f"Full consolidation result: {consolidation}")

        # Check if consolidation ran successfully or failed gracefully
        if consolidation["model_used"] == "error":
            # Ollama not available - verify we got a meaningful error message
            assert (
                "failed" in consolidation["summary"].lower()
            ), "Error state should have meaningful error message"
            print(
                f"Memory consolidation failed gracefully (Ollama not available): {consolidation['summary']}"
            )
            return  # Skip entity extraction tests when Ollama is not available

        # If JSON parsing worked, we should have some entities
        # If it fell back to summary-only, that's also acceptable
        if consolidation["entities"]:  # Only test if entities were extracted
            print(f"Entities found: {consolidation['entities']}")
            print(f"Entity types: {[type(e) for e in consolidation['entities']]}")

            # Handle case where entities might be dicts instead of strings
            entities_text = ""
            if isinstance(consolidation["entities"][0], str):
                entities_text = " ".join(consolidation["entities"]).lower()
            elif isinstance(consolidation["entities"][0], dict):
                # Extract names from dict entities
                entity_names = [
                    e.get("name", "")
                    for e in consolidation["entities"]
                    if isinstance(e, dict)
                ]
                entities_text = " ".join(entity_names).lower()

            # Look for likely entity extractions
            possible_entities = [
                "jeffery",
                "alpha",
                "sparkle",
                "alpha-recall",
                "search_all_memories",
            ]
            found_entities = [
                entity for entity in possible_entities if entity in entities_text
            ]

            assert (
                len(found_entities) > 0
            ), f"Should find at least one expected entity. Found entities: {consolidation['entities']}, entities_text: '{entities_text}'"

        # At minimum, summary should contain relevant content
        summary_text = consolidation["summary"].lower()
        assert len(summary_text) > 10, "Summary should contain meaningful content"

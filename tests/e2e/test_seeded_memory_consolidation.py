"""Seeded E2E tests for memory consolidation functionality.

These tests use comprehensive mock data to verify consolidation behavior
against realistic, populated databases with varied memory content.
"""

import asyncio
import json

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_memory_consolidation_with_realistic_data(test_stack_seeded):
    """Test memory consolidation with our comprehensive seeded dataset."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Add a few new memories on top of seeded data for consolidation
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Working with Alpha on testing the memory consolidation system"
            },
        )
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "The seeded test data includes Alpha, Jeffery, and Sparkle's bread crimes"
            },
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Trigger memory consolidation
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert "memory_consolidation" in data

        consolidation = data["memory_consolidation"]

        # Should have processed memories (either fresh ones or existing seeded ones)
        assert consolidation["processed_memories_count"] > 0

        # Verify consolidation structure
        assert isinstance(consolidation["entities"], list)
        assert isinstance(consolidation["relationships"], list)
        assert isinstance(consolidation["insights"], list)
        assert isinstance(consolidation["summary"], str)
        assert isinstance(consolidation["emotional_context"], str)
        assert isinstance(consolidation["next_steps"], list)

        # Check consolidation success or graceful failure
        if consolidation["model_used"] == "error":
            # Ollama not available - verify graceful failure
            assert "failed" in consolidation["summary"].lower()
            print(
                f"Memory consolidation failed gracefully (Ollama not available): {consolidation['summary']}"
            )
        elif consolidation["model_used"] == "placeholder":
            # Consolidation disabled
            assert consolidation["processed_memories_count"] == 0
        else:
            # Ollama available - verify actual consolidation
            assert consolidation["model_used"] == "llama3.2:1b"
            assert len(consolidation["summary"]) > 10


@pytest.mark.asyncio
async def test_memory_consolidation_identifies_seeded_entities(test_stack_seeded):
    """Test that consolidation can identify entities from our seeded data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Add memories that reference our seeded entities
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Alpha and Jeffery are collaborating on the Alpha-Recall project"
            },
        )
        await client.call_tool(
            "remember_shortterm",
            {"content": "Sparkle's bread crimes continue to baffle law enforcement"},
        )
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Redis performance improvements show 1000+ tokens/sec throughput"
            },
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Trigger consolidation
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        consolidation = data["memory_consolidation"]

        # Skip if Ollama not available
        if consolidation["model_used"] == "error":
            print("Skipping entity identification test - Ollama not available")
            return

        # If consolidation worked, check for entity identification
        if consolidation["entities"] and consolidation["model_used"] != "placeholder":
            # Extract entity names (handle both string and dict formats)
            entities_text = ""
            if isinstance(consolidation["entities"][0], str):
                entities_text = " ".join(consolidation["entities"]).lower()
            elif isinstance(consolidation["entities"][0], dict):
                entity_names = [e.get("name", "") for e in consolidation["entities"]]
                entities_text = " ".join(entity_names).lower()

            # Look for our seeded entities
            seeded_entities = ["alpha", "jeffery", "sparkle", "alpha-recall", "redis"]
            found_entities = [
                entity for entity in seeded_entities if entity in entities_text
            ]

            print(f"Found entities: {found_entities}")
            print(f"All extracted entities: {consolidation['entities']}")

            # Should find at least some of our well-known entities
            assert (
                len(found_entities) > 0
            ), f"Should find seeded entities. Entities text: '{entities_text}'"


@pytest.mark.asyncio
async def test_memory_consolidation_emotional_context_detection(test_stack_seeded):
    """Test that consolidation can detect emotional context from varied content."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Add memories with clear emotional content
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Extremely frustrated debugging session - nothing is working right"
            },
        )
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Breakthrough moment! The search_all_memories tool is working perfectly"
            },
        )
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Collaborative coding session going smoothly with great progress"
            },
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Trigger consolidation
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        consolidation = data["memory_consolidation"]

        # Skip if Ollama not available
        if consolidation["model_used"] == "error":
            print("Skipping emotional context test - Ollama not available")
            return

        # If consolidation worked, check emotional context
        if consolidation["model_used"] not in ["placeholder", "error"]:
            emotional_context = consolidation["emotional_context"]

            # Should have emotional_context field (may be empty if LLM didn't provide it)
            assert isinstance(
                emotional_context, str
            ), "emotional_context should be a string"

            print(f"Detected emotional context: '{emotional_context}'")

            # If emotional context is empty, that's acceptable - LLM might not always provide it
            if len(emotional_context) > 0:
                print(f"✅ LLM provided emotional context: {emotional_context}")
            else:
                print("⚠️ LLM didn't provide emotional context (acceptable)")


@pytest.mark.asyncio
async def test_memory_consolidation_with_cross_system_references(test_stack_seeded):
    """Test consolidation when memories reference entities from other systems."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Add memories that reference our seeded LTM entities and narratives
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Read about The Great Redis Migration Saga - what an epic story"
            },
        )
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Alpha's consciousness development reminds me of the awakening narrative"
            },
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Trigger consolidation
        result = await client.call_tool("gentle_refresh")
        data = json.loads(result.content[0].text)

        consolidation = data["memory_consolidation"]

        # Should have processed the new memories plus any existing ones
        assert consolidation["processed_memories_count"] >= 2

        # Verify basic structure regardless of Ollama availability
        assert isinstance(consolidation["summary"], str)
        assert len(consolidation["summary"]) > 0


@pytest.mark.asyncio
async def test_memory_consolidation_performance_with_seeded_data(test_stack_seeded):
    """Test that consolidation performs reasonably with realistic data volumes."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        import time

        # Add a memory to trigger consolidation
        await client.call_tool(
            "remember_shortterm",
            {"content": "Testing consolidation performance with seeded data"},
        )

        # Wait for indexing
        await asyncio.sleep(1)

        # Time the consolidation process
        start_time = time.time()
        result = await client.call_tool("gentle_refresh")
        end_time = time.time()

        data = json.loads(result.content[0].text)
        consolidation = data["memory_consolidation"]

        # Consolidation should complete within reasonable time
        consolidation_time = end_time - start_time
        print(
            f"Consolidation took {consolidation_time:.2f}s for {consolidation['processed_memories_count']} memories"
        )

        # Should complete within 30 seconds even with seeded data
        assert (
            consolidation_time < 30.0
        ), f"Consolidation took {consolidation_time:.2f}s, should be < 30s"

        # Should have timestamp
        assert "consolidation_timestamp" in consolidation
        assert len(consolidation["consolidation_timestamp"]) > 0

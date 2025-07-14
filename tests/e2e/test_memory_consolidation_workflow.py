"""E2E test for memory consolidation workflow.

This test validates the complete workflow of an AI agent reviewing short-term
memories and promoting important information to long-term storage.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.workflow_data_seeder import WorkflowDataSeeder


@pytest.mark.asyncio
class TestMemoryConsolidationWorkflow:
    """Test the complete memory consolidation workflow."""

    async def test_memory_consolidation_workflow(self, test_stack):
        """Test the complete workflow of reviewing and consolidating memories."""
        server_url = test_stack

        # Create fresh client and seed workflow data
        async with Client(server_url) as client:
            seeder = WorkflowDataSeeder(client)
            await seeder.seed_workflow_data("memory_consolidation")

            # Step 1: Verify initial state - check existing entities
            initial_entities = await client.call_tool("browse_longterm", {"limit": 100})
            initial_entity_data = json.loads(initial_entities.content[0].text)
            initial_entity_names = {
                e["entity_name"] for e in initial_entity_data["browse_data"]["entities"]
            }

            # Step 2: AI reviews recent memories to understand context
            recent_memories = await client.call_tool(
                "browse_shortterm", {"limit": 10, "order": "desc"}
            )

            # Parse the prose response to verify we have memories about Redis refactoring
            recent_memories_text = recent_memories.content[0].text
            assert "Redis refactoring" in recent_memories_text
            assert "god object" in recent_memories_text
            assert "Monday" in recent_memories_text

            # Step 3: AI identifies key concepts and creates entities
            # Create entity for the Redis refactoring project
            await client.call_tool(
                "remember_longterm",
                {
                    "entity": "Redis refactoring project",
                    "observation": "Major architectural improvement splitting a 1095-line god object into three focused services",
                    "type": "project",
                },
            )

            # Create entity for the Monday bug
            await client.call_tool(
                "remember_longterm",
                {
                    "entity": "Monday bug",
                    "observation": "Delightful test failure that only occurred on Mondays due to weekday() returning 0",
                    "type": "bug",
                },
            )

            # Step 4: AI creates relationships between entities
            await client.call_tool(
                "relate_longterm",
                {
                    "entity": "Redis refactoring project",
                    "to_entity": "Jeffery",
                    "as_type": "involves",
                },
            )

            await client.call_tool(
                "relate_longterm",
                {
                    "entity": "Monday bug",
                    "to_entity": "Redis refactoring project",
                    "as_type": "discovered_during",
                },
            )

            # Step 5: AI creates a narrative memory about the experience
            await client.call_tool(
                "remember_narrative",
                {
                    "title": "The Great Redis Refactoring and the Monday Mystery",
                    "paragraphs": [
                        "Today marked a significant milestone in cleaning up the Alpha-Recall codebase. Jeffery and I tackled a massive 1095-line Redis god object that had been making the code difficult to maintain.",
                        "We successfully split it into three focused services: RedisMemoryService, RedisIdentityService, and RedisContextService. Each service now has a clear, single responsibility, making the architecture much cleaner.",
                        "After the refactoring, we had to update all the unit tests. The imports needed changing, and the mocking strategies had to be adjusted, especially for gentle_refresh which uses all three services.",
                        "Then came the delightful surprise - a test that only failed on Mondays! It turned out the weekday() function returns 0 for Monday, but our test expected a range of 1-7. What a perfect ending to a productive refactoring session.",
                    ],
                    "participants": ["Alpha", "Jeffery"],
                    "outcome": "breakthrough",
                    "tags": ["refactoring", "testing", "debugging", "architecture"],
                },
            )

            # Step 6: Verify the workflow created expected outcomes
            # Check that new entities were created
            final_entities = await client.call_tool("browse_longterm", {"limit": 100})
            final_entity_data = json.loads(final_entities.content[0].text)
            final_entity_names = {
                e["entity_name"] for e in final_entity_data["browse_data"]["entities"]
            }

            new_entities = final_entity_names - initial_entity_names
            assert "Redis refactoring project" in new_entities
            assert "Monday bug" in new_entities

            # Check relationships were created
            redis_project = await client.call_tool(
                "get_relationships", {"entity_name": "Redis refactoring project"}
            )
            relationships = json.loads(redis_project.content[0].text)

            # Verify expected relationships exist
            involves_jeffery = any(
                r["to_entity"] == "Jeffery" and r["type"] == "involves"
                for r in relationships["outgoing"]
            )
            assert involves_jeffery, "Redis project should involve Jeffery"

            discovered_bug = any(
                r["from_entity"] == "Monday bug" and r["type"] == "discovered_during"
                for r in relationships["incoming"]
            )
            assert (
                discovered_bug
            ), "Monday bug should be discovered during Redis project"

            # Search for the narrative to verify it was stored
            narrative_search = await client.call_tool(
                "search_narratives", {"query": "Redis refactoring Monday", "limit": 5}
            )
            search_results = json.loads(narrative_search.content[0].text)

            assert len(search_results["results"]) > 0, "Should find the narrative"
            assert any(
                "Great Redis Refactoring" in r["title"]
                for r in search_results["results"]
            ), "Should find our specific narrative"

            # Step 7: Verify semantic connections work
            # Search across all memories for "Monday bug"
            unified_search = await client.call_tool(
                "search_all_memories", {"query": "Monday bug test failure", "limit": 10}
            )

            # Should find results across multiple memory types
            unified_search_text = unified_search.content[0].text
            assert "Monday" in unified_search_text
            assert "weekday()" in unified_search_text

            # The workflow is complete - memories have been reviewed,
            # important concepts promoted to LTM, relationships established,
            # and a narrative created to capture the experience

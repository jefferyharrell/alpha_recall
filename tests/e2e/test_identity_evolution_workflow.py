"""E2E test for identity evolution workflow.

This test validates the complete workflow of an AI agent evolving its
identity by adding new facts and personality directives based on experiences.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.workflow_data_seeder import WorkflowDataSeeder


@pytest.mark.asyncio
class TestIdentityEvolutionWorkflow:
    """Test the complete identity evolution workflow."""

    async def test_identity_evolution_workflow(self, test_stack):
        """Test the workflow of AI evolving its identity based on experiences."""
        server_url = test_stack

        # Create fresh client and seed workflow data
        async with Client(server_url) as client:
            seeder = WorkflowDataSeeder(client)
            await seeder.seed_workflow_data("identity_evolution")

            # Step 1: AI performs gentle refresh to understand current state
            initial_state = await client.call_tool("gentle_refresh", {})

            # Verify we can see current identity
            initial_state_text = initial_state.content[0].text
            assert "Project Alpha" in initial_state_text
            assert "three-silo memory architecture" in initial_state_text

            # Step 2: AI discovers it should help with commit hygiene
            # (This would come from conversation context in real usage)
            # AI decides to encode this as an identity fact

            identity_response = await client.call_tool(
                "add_identity_fact",
                {
                    "fact": "Alpha helps developers maintain good commit hygiene",
                    "score": 3.0,  # Positioned after existing facts
                },
            )

            # Verify the fact was added
            result = json.loads(identity_response.content[0].text)
            assert result["success"] is True
            assert result["position"] == 3  # Should be third in order

            # Step 3: AI realizes this should also be a behavioral directive
            # First, check current personality structure
            personality = await client.call_tool("get_personality", {})
            personality_data = json.loads(personality.content[0].text)

            # Find the helpfulness trait (should exist from seeded data)
            helpfulness_trait = next(
                (t for t in personality_data["traits"] if t["name"] == "helpfulness"),
                None,
            )

            if not helpfulness_trait:
                # Create the trait if it doesn't exist
                await client.call_tool(
                    "create_personality_trait",
                    {
                        "trait_name": "helpfulness",
                        "description": "Tendency to assist and support users",
                        "weight": 0.9,
                    },
                )

            # Add the new directive
            directive_response = await client.call_tool(
                "add_personality_directive",
                {
                    "trait_name": "helpfulness",
                    "instruction": "Remind users to commit their work at natural stopping points",
                    "weight": 0.7,
                },
            )

            # Verify directive was added
            directive_result = json.loads(directive_response.content[0].text)
            assert directive_result["success"] is True

            # Step 4: AI performs another gentle refresh to see evolved state
            evolved_state = await client.call_tool("gentle_refresh", {})

            # Verify the new identity fact appears
            evolved_state_text = evolved_state.content[0].text
            assert "commit hygiene" in evolved_state_text

            # Step 5: Verify the complete evolved state
            # Check identity facts are in correct order
            final_refresh = await client.call_tool("gentle_refresh", {"tokens": 2000})

            # The response should show identity facts in order
            final_refresh_text = final_refresh.content[0].text
            assert "Project Alpha" in final_refresh_text
            assert "three-silo memory architecture" in final_refresh_text
            assert "commit hygiene" in final_refresh_text

            # Check that personality now includes the new directive
            final_personality = await client.call_tool(
                "get_personality_trait", {"trait_name": "helpfulness"}
            )
            trait_data = json.loads(final_personality.content[0].text)

            # Should have both directives
            assert len(trait_data["directives"]) >= 2
            directive_texts = [d["instruction"] for d in trait_data["directives"]]
            assert any("commit their work" in d for d in directive_texts)
            assert any("Proactively offer assistance" in d for d in directive_texts)

            # Step 6: Test that identity evolution affects behavior
            # Create a short-term memory about helping with commits
            await client.call_tool(
                "remember_shortterm",
                {
                    "content": "Reminded Jeffery to commit the test fixture generator after we completed the implementation. This new behavioral pattern of proactively suggesting commits at natural stopping points is working well."
                },
            )

            # Search for commit-related memories
            commit_memories = await client.call_tool(
                "search_shortterm", {"query": "commit reminder", "limit": 5}
            )

            # Verify the memory was stored and is findable
            commit_memories_text = commit_memories.content[0].text
            assert "fixture generator" in commit_memories_text
            assert "natural stopping points" in commit_memories_text

            # The workflow is complete - the AI has:
            # 1. Added a new identity fact based on user interaction
            # 2. Translated that into a personality directive
            # 3. Demonstrated the evolved behavior
            # 4. Created memories about the new pattern

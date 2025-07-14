"""Data seeding for workflow-based E2E tests using generated fixtures."""

import json
from pathlib import Path
from typing import Any

from fastmcp import Client


class WorkflowDataSeeder:
    """Seeds test databases with workflow-specific test data."""

    def __init__(self, client: Client):
        """Initialize with MCP client."""
        self.client = client
        self.fixture_dir = Path(__file__).parent.parent.parent / "fixtures/generated"

    async def seed_workflow_data(self, workflow_name: str) -> dict[str, Any]:
        """Seed data for a specific workflow test.

        Args:
            workflow_name: Name of the workflow (e.g., "memory_consolidation", "identity_evolution")

        Returns:
            Dictionary with the loaded test data
        """
        fixture_path = self.fixture_dir / f"{workflow_name}_workflow.json"

        if not fixture_path.exists():
            raise FileNotFoundError(
                f"Workflow fixture not found: {fixture_path}. "
                f"Run 'just generate-test-fixtures' to create it."
            )

        with open(fixture_path) as f:
            data = json.load(f)

        # Seed initial state based on workflow type
        if workflow_name == "memory_consolidation":
            await self._seed_memory_consolidation(data)
        elif workflow_name == "identity_evolution":
            await self._seed_identity_evolution(data)
        else:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        return data

    async def _seed_memory_consolidation(self, data: dict[str, Any]) -> None:
        """Seed initial state for memory consolidation workflow."""
        # Seed short-term memories with pre-computed embeddings
        for memory in data["initial_state"]["memories"]:
            # Store directly in Redis with embeddings
            # Note: In a real implementation, we'd use Redis commands directly
            # For now, we'll use the MCP tools which will compute embeddings again
            # This is okay for testing since we're validating workflow, not embeddings
            await self.client.call_tool(
                "remember_shortterm", {"content": memory["content"]}
            )

        # Seed initial entities
        for entity in data["initial_state"]["entities"]:
            for observation in entity["observations"]:
                # Extract text from observation (it's now a dict with embeddings)
                obs_text = (
                    observation["text"]
                    if isinstance(observation, dict)
                    else observation
                )
                await self.client.call_tool(
                    "remember_longterm",
                    {
                        "entity": entity["name"],
                        "observation": obs_text,
                        "type": entity.get("type"),
                    },
                )

    async def _seed_identity_evolution(self, data: dict[str, Any]) -> None:
        """Seed initial state for identity evolution workflow."""
        # Seed identity facts
        for fact in data["initial_state"]["identity_facts"]:
            await self.client.call_tool(
                "add_identity_fact", {"fact": fact["fact"], "score": fact["score"]}
            )

        # Seed personality traits
        for trait in data["initial_state"]["personality_traits"]:
            # Create trait
            await self.client.call_tool(
                "create_personality_trait",
                {
                    "trait_name": trait["name"],
                    "description": trait["description"],
                    "weight": trait["weight"],
                },
            )

            # Add directives
            for directive in trait.get("directives", []):
                await self.client.call_tool(
                    "add_personality_directive",
                    {
                        "trait_name": trait["name"],
                        "instruction": directive["instruction"],
                        "weight": directive["weight"],
                    },
                )

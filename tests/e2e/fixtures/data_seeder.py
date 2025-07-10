"""Data seeding infrastructure for E2E tests with predictable mock data."""

import asyncio
import json
from pathlib import Path
from typing import Any

from fastmcp import Client


class TestDataSeeder:
    """Seeds Alpha-Recall test databases with comprehensive mock data."""

    def __init__(self, client: Client):
        """Initialize with MCP client for tool calls."""
        self.client = client
        self.mock_data_dir = Path(__file__).parent.parent.parent.parent / "mock_data"

    async def seed_all_data(self) -> dict[str, Any]:
        """Seed all test data and return mapping for test assertions.

        Returns:
            Dictionary with seeded data references for predictable testing:
            {
                'stm': {'memory_ids': [...], 'count': 20},
                'ltm': {'entities': {...}, 'relationships': [...]},
                'nm': {'story_ids': [...], 'narratives': {...}}
            }
        """
        print("🌱 Seeding comprehensive test data...")

        # Seed in logical order: LTM entities first, then STM, then NM narratives
        ltm_data = await self.seed_ltm_data()
        stm_data = await self.seed_stm_data()
        nm_data = await self.seed_nm_data()

        seeded_data = {
            "stm": stm_data,
            "ltm": ltm_data,
            "nm": nm_data,
            "total_memories": stm_data["count"],
            "total_entities": len(ltm_data["entities"]),
            "total_narratives": len(nm_data["story_ids"]),
        }

        print(
            f"✅ Seeding complete: {seeded_data['total_memories']} STM, {seeded_data['total_entities']} entities, {seeded_data['total_narratives']} narratives"
        )
        return seeded_data

    async def seed_stm_data(self) -> dict[str, Any]:
        """Seed short-term memory data.

        Returns:
            {
                'memory_ids': [list of stored memory IDs],
                'count': total_count,
                'categories': {category: [memory_ids]},
                'emotional_tones': {tone: [memory_ids]}
            }
        """
        print("📝 Seeding STM data...")

        stm_file = self.mock_data_dir / "stm_test_data.json"
        with open(stm_file) as f:
            stm_mock_data = json.load(f)

        memory_ids = []
        categories = {}
        emotional_tones = {}

        for memory in stm_mock_data["memories"]:
            # Store the memory
            result = await self.client.call_tool(
                "remember_shortterm", {"content": memory["content"]}
            )

            response_data = json.loads(result.content[0].text)
            memory_id = response_data["memory_id"]
            memory_ids.append(memory_id)

            # Track by category for targeted testing
            category = memory["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(memory_id)

            # Track by emotional tone for clustering tests
            tone = memory["emotional_tone"]
            if tone not in emotional_tones:
                emotional_tones[tone] = []
            emotional_tones[tone].append(memory_id)

            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)

        return {
            "memory_ids": memory_ids,
            "count": len(memory_ids),
            "categories": categories,
            "emotional_tones": emotional_tones,
        }

    async def seed_ltm_data(self) -> dict[str, Any]:
        """Seed long-term memory entities and relationships.

        Returns:
            {
                'entities': {entity_name: entity_info},
                'relationships': [relationship_list],
                'entity_names': [list of entity names]
            }
        """
        print("🧠 Seeding LTM data...")

        ltm_file = self.mock_data_dir / "ltm_test_data.json"
        with open(ltm_file) as f:
            ltm_mock_data = json.load(f)

        entities = {}

        # First, create all entities with their observations
        for entity_data in ltm_mock_data["entities"]:
            entity_name = entity_data["name"]
            entity_type = entity_data["type"]

            # Create the entity
            await self.client.call_tool(
                "remember_longterm",
                {
                    "entity": entity_name,
                    "type": entity_type,
                    "observation": f"Test entity of type {entity_type}",
                },
            )

            # Add all observations for this entity
            for observation in entity_data["observations"]:
                await self.client.call_tool(
                    "remember_longterm",
                    {"entity": entity_name, "observation": observation},
                )
                await asyncio.sleep(0.05)  # Gentle pacing

            entities[entity_name] = {
                "type": entity_type,
                "observation_count": len(entity_data["observations"]),
            }

        # Then create relationships
        relationships = []
        for rel_data in ltm_mock_data["relationships"]:
            await self.client.call_tool(
                "relate_longterm",
                {
                    "entity": rel_data["from"],
                    "to_entity": rel_data["to"],
                    "as_type": rel_data["type"],
                },
            )
            relationships.append(rel_data)
            await asyncio.sleep(0.05)

        return {
            "entities": entities,
            "relationships": relationships,
            "entity_names": list(entities.keys()),
        }

    async def seed_nm_data(self) -> dict[str, Any]:
        """Seed narrative memory stories.

        Returns:
            {
                'story_ids': [list of story IDs],
                'narratives': {story_id: narrative_info},
                'participants': {participant: [story_ids]},
                'outcomes': {outcome: [story_ids]}
            }
        """
        print("📚 Seeding narrative memory data...")

        nm_file = self.mock_data_dir / "nm_test_data.json"
        with open(nm_file) as f:
            nm_mock_data = json.load(f)

        story_ids = []
        narratives = {}
        participants = {}
        outcomes = {}

        for narrative in nm_mock_data["narratives"]:
            # Store the narrative
            result = await self.client.call_tool(
                "remember_narrative",
                {
                    "title": narrative["title"],
                    "paragraphs": narrative["paragraphs"],
                    "participants": narrative["participants"],
                    "outcome": narrative["outcome"],
                    "tags": narrative.get("tags", []),
                },
            )

            response_data = json.loads(result.content[0].text)
            story_id = response_data["story"]["story_id"]
            story_ids.append(story_id)

            # Track narrative info
            narratives[story_id] = {
                "title": narrative["title"],
                "participant_count": len(narrative["participants"]),
                "paragraph_count": len(narrative["paragraphs"]),
                "outcome": narrative["outcome"],
            }

            # Track by participants for targeted testing
            for participant in narrative["participants"]:
                if participant not in participants:
                    participants[participant] = []
                participants[participant].append(story_id)

            # Track by outcome for filtering tests
            outcome = narrative["outcome"]
            if outcome not in outcomes:
                outcomes[outcome] = []
            outcomes[outcome].append(story_id)

            await asyncio.sleep(0.2)  # Narratives are more complex, need more time

        return {
            "story_ids": story_ids,
            "narratives": narratives,
            "participants": participants,
            "outcomes": outcomes,
        }


# Convenience fixture for easy test access
async def seed_test_data(client: Client) -> dict[str, Any]:
    """Convenience function to seed all test data.

    Args:
        client: MCP client for tool calls

    Returns:
        Complete seeded data mapping for test assertions
    """
    seeder = TestDataSeeder(client)
    return await seeder.seed_all_data()

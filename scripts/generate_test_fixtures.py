#!/usr/bin/env python3
"""Generate test fixtures with pre-computed embeddings using current models.

This script:
1. Reads source test data (text only) from tests/fixtures/source_data/
2. Computes embeddings using the currently configured models
3. Saves complete fixtures to tests/fixtures/generated/

This allows test data to adapt to model changes without storing embeddings in git.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_recall.config import settings
from alpha_recall.logging import get_logger
from alpha_recall.services.embedding import EmbeddingService

logger = get_logger(__name__)


class TestFixtureGenerator:
    """Generates test fixtures with pre-computed embeddings."""

    def __init__(self):
        """Initialize with embedding service."""
        self.embedding_service = EmbeddingService()
        self.source_dir = Path(__file__).parent.parent / "tests/fixtures/source_data"
        self.output_dir = Path(__file__).parent.parent / "tests/fixtures/generated"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_all_fixtures(self) -> None:
        """Generate fixtures for all source files."""
        logger.info(
            "Generating test fixtures",
            semantic_model=settings.semantic_embedding_model,
            emotional_model=settings.emotional_embedding_model,
        )

        # Process each source file
        source_files = list(self.source_dir.glob("*.json"))
        if not source_files:
            logger.warning("No source files found", path=str(self.source_dir))
            return

        for source_file in source_files:
            await self._process_fixture(source_file)

        logger.info(
            "Test fixture generation complete",
            fixtures_generated=len(source_files),
            output_dir=str(self.output_dir),
        )

    async def _process_fixture(self, source_file: Path) -> None:
        """Process a single fixture file."""
        logger.info("Processing fixture", source_file=source_file.name)

        # Load source data
        with open(source_file) as f:
            data = json.load(f)

        # Generate embeddings based on workflow type
        if data["workflow"] == "memory_consolidation":
            await self._process_memory_consolidation(data)
        elif data["workflow"] == "identity_evolution":
            await self._process_identity_evolution(data)
        elif data["workflow"] == "cross_session_continuity":
            await self._process_cross_session(data)
        else:
            logger.warning("Unknown workflow type", workflow=data.get("workflow"))
            return

        # Save to output
        output_file = self.output_dir / source_file.name
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Fixture generated", output_file=output_file.name)

    async def _process_memory_consolidation(self, data: dict[str, Any]) -> None:
        """Add embeddings to memory consolidation workflow data."""
        # Process memories
        for memory in data["initial_state"]["memories"]:
            content = memory["content"]

            # Generate embeddings
            semantic_embedding = self.embedding_service.encode_semantic(content)
            emotional_embedding = self.embedding_service.encode_emotional(content)

            # Store as lists (JSON serializable)
            memory["semantic_embedding"] = (
                semantic_embedding
                if isinstance(semantic_embedding, list)
                else semantic_embedding.tolist()
            )
            memory["emotional_embedding"] = (
                emotional_embedding
                if isinstance(emotional_embedding, list)
                else emotional_embedding.tolist()
            )

            # Add metadata
            memory["semantic_dims"] = len(semantic_embedding)
            memory["emotional_dims"] = len(emotional_embedding)

        # Process entity observations
        for entity in data["initial_state"]["entities"]:
            for i, observation in enumerate(entity["observations"]):
                # Generate semantic embedding for observations
                semantic_embedding = self.embedding_service.encode_semantic(observation)

                # Convert observations to dict format with embeddings
                entity["observations"][i] = {
                    "text": observation,
                    "semantic_embedding": (
                        semantic_embedding
                        if isinstance(semantic_embedding, list)
                        else semantic_embedding.tolist()
                    ),
                    "semantic_dims": len(semantic_embedding),
                }

    async def _process_identity_evolution(self, data: dict[str, Any]) -> None:
        """Add embeddings to identity evolution workflow data."""
        # Identity facts don't need embeddings (stored as-is in Redis)
        # Personality traits don't need embeddings either
        # This workflow tests non-embedding functionality
        logger.info("Identity evolution workflow - no embeddings needed")

    async def _process_cross_session(self, data: dict[str, Any]) -> None:
        """Add embeddings to cross-session continuity workflow data."""
        # Process any memories in the workflow
        if "memories" in data.get("initial_state", {}):
            for memory in data["initial_state"]["memories"]:
                content = memory["content"]

                semantic_embedding = self.embedding_service.encode_semantic(content)
                emotional_embedding = self.embedding_service.encode_emotional(content)

                memory["semantic_embedding"] = (
                    semantic_embedding
                    if isinstance(semantic_embedding, list)
                    else semantic_embedding.tolist()
                )
                memory["emotional_embedding"] = (
                    emotional_embedding
                    if isinstance(emotional_embedding, list)
                    else emotional_embedding.tolist()
                )
                memory["semantic_dims"] = len(semantic_embedding)
                memory["emotional_dims"] = len(emotional_embedding)


async def main():
    """Main entry point."""
    print("🔧 Generating test fixtures...")
    print(f"📊 Semantic model: {settings.semantic_embedding_model}")
    print(f"💭 Emotional model: {settings.emotional_embedding_model}")
    print()

    generator = TestFixtureGenerator()
    await generator.generate_all_fixtures()

    print()
    print("✅ Test fixtures generated successfully!")
    print("📁 Output directory: tests/fixtures/generated/")


if __name__ == "__main__":
    asyncio.run(main())

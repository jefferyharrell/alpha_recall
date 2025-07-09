"""Memory consolidation service using Ollama."""

from datetime import UTC
from typing import Any

import ollama
from rich import print as rich_print

from ..config import settings
from ..logging import get_logger
from ..services.redis import get_redis_service
from ..services.template_loader import template_loader
from ..utils.correlation import generate_correlation_id, set_correlation_id

logger = get_logger(__name__)


class MemoryConsolidationService:
    """Service for consolidating short-term memories using Ollama."""

    def __init__(self):
        """Initialize the memory consolidation service."""
        self.redis_service = get_redis_service()

    async def consolidate_memories(self) -> dict[str, Any]:
        """Consolidate recent short-term memories using Ollama.

        Returns:
            Dictionary containing consolidation results
        """
        correlation_id = generate_correlation_id("consolidate")
        set_correlation_id(correlation_id)

        logger.info("Starting memory consolidation", correlation_id=correlation_id)

        try:
            # Check if consolidation is enabled
            if not settings.memory_consolidation_enabled:
                logger.debug(
                    "Memory consolidation disabled", correlation_id=correlation_id
                )
                return self._empty_consolidation_result()

            # Get recent memories from Redis
            memories = await self._get_recent_memories()

            if not memories:
                logger.debug("No recent memories found", correlation_id=correlation_id)
                return self._empty_consolidation_result()

            # Render the prompt template
            prompt = self._render_consolidation_prompt(memories)

            # Call Ollama for consolidation
            logger.info(
                f"Consolidating {len(memories)} memories with {settings.helper_model}",
                correlation_id=correlation_id,
            )

            try:
                # Create Ollama client configured for our host/port
                client = ollama.Client(
                    host=f"http://{settings.consolidation_ollama_host}:{settings.consolidation_ollama_port}"
                )

                # Print the prompt being sent to the helper model
                rich_print(prompt)

                # Generate response from helper model
                response = client.generate(
                    model=settings.helper_model, prompt=prompt, stream=False
                )

                helper_response = response["response"]

                # Print the helper model response to stdout with Rich
                rich_print(helper_response)

                logger.info(
                    "Memory consolidation completed successfully",
                    correlation_id=correlation_id,
                )

                return {
                    "entities": [],
                    "relationships": [],
                    "insights": [],
                    "summary": helper_response,
                    "emotional_context": "",
                    "next_steps": [],
                    "processed_memories_count": len(memories),
                    "consolidation_timestamp": self._get_timestamp(),
                    "model_used": settings.helper_model,
                }

            except Exception as e:
                logger.error(
                    f"Ollama consolidation failed: {e}", correlation_id=correlation_id
                )
                # Return a fallback response
                return {
                    "entities": [],
                    "relationships": [],
                    "insights": [],
                    "summary": f"Memory consolidation failed: {str(e)}",
                    "emotional_context": "",
                    "next_steps": [],
                    "processed_memories_count": len(memories),
                    "consolidation_timestamp": self._get_timestamp(),
                    "model_used": "error",
                }

        except Exception as e:
            logger.error(
                f"Memory consolidation failed: {e}", correlation_id=correlation_id
            )
            return self._empty_consolidation_result()

    async def _get_recent_memories(self) -> list[dict[str, Any]]:
        """Get recent memories from Redis.

        Returns:
            List of memory dictionaries with content and timestamp
        """
        try:
            # Get memory IDs from the last 24 hours
            # For now, just get the 10 most recent memories
            memory_ids_with_scores = self.redis_service.client.zrevrange(
                "memory_index", 0, 9, withscores=True
            )

            memories = []
            for memory_id_bytes, _timestamp in memory_ids_with_scores:
                memory_id = memory_id_bytes.decode("utf-8")
                memory_key = f"memory:{memory_id}"

                # Get memory data from hash using the same pattern as gentle_refresh
                memory_data = self.redis_service.client.hmget(
                    memory_key, ["content", "created_at"]
                )

                if memory_data[0] is not None:  # Content exists
                    content = memory_data[0].decode("utf-8")
                    created_at = (
                        memory_data[1].decode("utf-8") if memory_data[1] else ""
                    )

                    # Extract just content and timestamp for consolidation
                    memories.append(
                        {
                            "content": content,
                            "timestamp": created_at,
                        }
                    )

            logger.debug(f"Retrieved {len(memories)} recent memories")
            return memories

        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    def _render_consolidation_prompt(self, memories: list[dict[str, Any]]) -> str:
        """Render the consolidation prompt template.

        Args:
            memories: List of memory dictionaries

        Returns:
            Rendered prompt string
        """
        try:
            context = {
                "memories": memories,
                "memory_count": len(memories),
                "model": settings.helper_model,
            }

            prompt = template_loader.render_template(
                "memory_consolidation.md.j2", context
            )
            logger.debug("Rendered consolidation prompt")
            return prompt

        except Exception as e:
            logger.error(f"Failed to render consolidation prompt: {e}")
            return "Please say 'rubber baby buggy bumpers.'"

    def _empty_consolidation_result(self) -> dict[str, Any]:
        """Return an empty consolidation result.

        Returns:
            Empty consolidation result dictionary
        """
        return {
            "entities": [],
            "relationships": [],
            "insights": [],
            "summary": "",
            "emotional_context": "",
            "next_steps": [],
            "processed_memories_count": 0,
            "consolidation_timestamp": self._get_timestamp(),
            "model_used": "placeholder",
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format.

        Returns:
            ISO timestamp string
        """
        from datetime import datetime

        return datetime.now(UTC).isoformat()


# Global consolidation service instance
consolidation_service = MemoryConsolidationService()

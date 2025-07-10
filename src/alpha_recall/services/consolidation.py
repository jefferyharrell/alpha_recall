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
                f"Consolidating {len(memories)} memories with {settings.helper_model} (timeout: {settings.consolidation_timeout}s)",
                correlation_id=correlation_id,
            )

            try:
                # Create Ollama client configured for our host/port
                client = ollama.Client(
                    host=f"http://{settings.consolidation_ollama_host}:{settings.consolidation_ollama_port}"
                )

                # Print the prompt being sent to the helper model (truncated to 100 lines)
                rich_print(self._truncate_prompt_for_display(prompt))

                # Generate response from helper model with timeout
                response = client.generate(
                    model=settings.helper_model,
                    prompt=prompt,
                    stream=False,
                    options={"timeout": settings.consolidation_timeout},
                )

                helper_response = response["response"]

                # Print the helper model response to stdout with Rich
                rich_print(helper_response)

                # Try to parse the response as structured JSON
                try:
                    import json

                    # Extract JSON from response (handle cases where model adds extra text)
                    json_start = helper_response.find("{")
                    json_end = helper_response.rfind("}") + 1

                    if json_start != -1 and json_end > json_start:
                        json_text = helper_response[json_start:json_end]
                        parsed_result = json.loads(json_text)

                        logger.info(
                            "Successfully parsed structured consolidation response",
                            correlation_id=correlation_id,
                        )

                        # Return structured result with parsed fields
                        return {
                            "entities": parsed_result.get("entities", []),
                            "relationships": parsed_result.get("relationships", []),
                            "insights": parsed_result.get("insights", []),
                            "summary": parsed_result.get("summary", ""),
                            "emotional_context": parsed_result.get(
                                "emotional_context", ""
                            ),
                            "next_steps": parsed_result.get("next_steps", []),
                            "processed_memories_count": len(memories),
                            "consolidation_timestamp": self._get_timestamp(),
                            "model_used": settings.helper_model,
                        }
                    else:
                        raise json.JSONDecodeError("No JSON found in response", "", 0)

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse structured response as JSON: {e}. Using fallback.",
                        correlation_id=correlation_id,
                    )
                    # Fallback to putting everything in summary
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

                logger.info(
                    "Memory consolidation completed successfully",
                    correlation_id=correlation_id,
                )

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
        """Get recent memories from Redis within the configured time window.

        Returns:
            List of memory dictionaries with content and timestamp
        """
        try:
            # Parse the time window (e.g., "24h" -> 24 hours)
            time_window = self._parse_time_window(settings.consolidation_time_window)
            cutoff_timestamp = self._get_cutoff_timestamp(time_window)

            logger.debug(f"Retrieving memories since {cutoff_timestamp}")

            # Get memory IDs from the sorted set (score is timestamp)
            # Use zrevrangebyscore to get memories within time window
            memory_ids_with_scores = self.redis_service.client.zrevrangebyscore(
                "memory_index",
                max="+inf",  # No upper bound
                min=cutoff_timestamp,  # Only memories after cutoff
                withscores=True,
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

            logger.info(
                f"Retrieved {len(memories)} memories from last {settings.consolidation_time_window}"
            )
            logger.debug(
                f"Memory IDs found: {len(memory_ids_with_scores)} total, {len(memories)} valid"
            )
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
                "consolidation_time_window": settings.consolidation_time_window,
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

    def _truncate_prompt_for_display(self, prompt: str, max_lines: int = 100) -> str:
        """Truncate prompt to max_lines for display purposes.

        Args:
            prompt: The full prompt text
            max_lines: Maximum number of lines to show (default: 100)

        Returns:
            Truncated prompt with truncation message if needed
        """
        lines = prompt.split("\n")
        if len(lines) <= max_lines:
            return prompt

        truncated_lines = lines[:max_lines]
        truncated_lines.append(
            f"\n... [TRUNCATED: {len(lines) - max_lines} more lines] ..."
        )
        return "\n".join(truncated_lines)

    def _parse_time_window(self, time_window: str) -> int:
        """Parse time window string to seconds.

        Args:
            time_window: Time window string like "24h", "30m", "7d"

        Returns:
            Time window in seconds
        """
        import re

        # Match pattern like "24h", "30m", "7d"
        match = re.match(r"^(\d+)([hmd])$", time_window.lower())
        if not match:
            logger.warning(
                f"Invalid time window format: {time_window}, defaulting to 24h"
            )
            return 24 * 3600  # 24 hours default

        value, unit = match.groups()
        value = int(value)

        if unit == "h":
            return value * 3600  # hours to seconds
        elif unit == "m":
            return value * 60  # minutes to seconds
        elif unit == "d":
            return value * 86400  # days to seconds
        else:
            return 24 * 3600  # default to 24 hours

    def _get_cutoff_timestamp(self, time_window_seconds: int) -> float:
        """Get the cutoff timestamp for memory retrieval.

        Args:
            time_window_seconds: Time window in seconds

        Returns:
            Unix timestamp for cutoff
        """
        from datetime import datetime

        now = datetime.now(UTC)
        cutoff_time = now.timestamp() - time_window_seconds
        return cutoff_time


# Global consolidation service instance
consolidation_service = MemoryConsolidationService()

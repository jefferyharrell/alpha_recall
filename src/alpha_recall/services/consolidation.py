"""Memory consolidation service with conversational reflection approach."""

import time
from datetime import UTC, datetime
from typing import Any

import ollama
from pydantic import ValidationError

from ..config import settings
from ..logging import get_logger
from ..schemas.consolidation import ShortTermMemory
from ..services.redis import get_redis_service
from ..services.template_loader import template_loader
from ..utils.correlation import generate_correlation_id, set_correlation_id

logger = get_logger(__name__)


class ConsolidationService:
    """Memory consolidation service using conversational reflection approach."""

    def __init__(self):
        """Initialize the consolidation service."""
        self.redis_service = get_redis_service()

    async def consolidate_shortterm_memories(
        self,
        time_window: str = "24h",
        model_name: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Consolidate short-term memories using conversational reflection.

        Args:
            time_window: Time window for memory retrieval (e.g., "24h", "7d", "30m")
            model_name: Override default helper model for testing
            temperature: Model temperature (default 0.0 for deterministic testing)

        Returns:
            Dictionary containing narrative reflection or failure information
        """
        correlation_id = generate_correlation_id("consolidate")
        set_correlation_id(correlation_id)

        logger.info(
            "Starting memory consolidation",
            time_window=time_window,
            model_name=model_name,
            temperature=temperature,
            correlation_id=correlation_id,
        )

        start_time = time.time()

        try:
            # Check if consolidation is enabled
            if not settings.memory_consolidation_enabled:
                logger.debug(
                    "Memory consolidation disabled", correlation_id=correlation_id
                )
                return self._disabled_response()

            # Get recent memories and validate input
            memories = await self._get_recent_memories_for_consolidation(time_window)

            if not memories:
                logger.debug("No recent memories found", correlation_id=correlation_id)
                return self._empty_response()

            # Call helper model for conversational consolidation
            model_to_use = model_name or settings.helper_model
            prompt = self._get_conversational_prompt(memories, time_window)

            narrative_response = await self._call_helper_model(
                prompt,
                model_to_use,
                temperature,
                correlation_id,
            )

            if narrative_response is None:
                return self._model_error_response("Helper model call failed")

            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Conversational memory consolidation completed",
                model=model_to_use,
                processing_time_ms=processing_time_ms,
                response_length=len(narrative_response),
                correlation_id=correlation_id,
            )

            return {
                "success": True,
                "narrative": narrative_response.strip(),
                "metadata": {
                    "processing_time_ms": processing_time_ms,
                    "model_name": model_to_use,
                    "temperature": temperature,
                    "input_memories_count": len(memories),
                    "time_window": time_window,
                    "response_length": len(narrative_response),
                    "approach": "conversational_reflection",
                },
                "correlation_id": correlation_id,
            }

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Memory consolidation failed with exception",
                error=str(e),
                error_type=type(e).__name__,
                processing_time_ms=processing_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "success": False,
                "error": f"Consolidation failed: {str(e)}",
                "error_type": type(e).__name__,
                "metadata": {
                    "processing_time_ms": processing_time_ms,
                    "model_name": model_name or settings.helper_model,
                    "temperature": temperature,
                    "approach": "conversational_reflection",
                },
                "correlation_id": correlation_id,
            }

    async def _get_recent_memories_for_consolidation(
        self, time_window: str
    ) -> list[ShortTermMemory]:
        """Get recent memories formatted for consolidation input validation."""
        try:
            # Parse time window and get memories (reuse existing logic)
            time_window_seconds = self._parse_time_window(time_window)
            cutoff_timestamp = self._get_cutoff_timestamp(time_window_seconds)

            # Get memory IDs from the sorted set
            memory_ids_with_scores = self.redis_service.client.zrevrangebyscore(
                "memory_index",
                max="+inf",
                min=cutoff_timestamp,
                withscores=True,
            )

            memories = []
            for memory_id_bytes, _timestamp in memory_ids_with_scores:
                memory_id = memory_id_bytes.decode("utf-8")
                memory_key = f"memory:{memory_id}"

                memory_data = self.redis_service.client.hmget(
                    memory_key, ["content", "created_at"]
                )

                if memory_data[0] is not None:
                    content = memory_data[0].decode("utf-8")
                    created_at = (
                        memory_data[1].decode("utf-8") if memory_data[1] else ""
                    )

                    # Create ShortTermMemory objects for validation
                    try:
                        memory = ShortTermMemory(content=content, timestamp=created_at)
                        memories.append(memory)
                    except ValidationError as e:
                        logger.warning(
                            f"Skipping invalid memory {memory_id}: {e}",
                        )

            logger.info(f"Retrieved {len(memories)} valid memories for consolidation")
            return memories

        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    def _get_conversational_prompt(
        self, memories: list[ShortTermMemory], time_window: str
    ) -> str:
        """Render the conversational consolidation prompt template."""
        try:
            context = {
                "memories": [
                    {"content": m.content, "timestamp": m.timestamp} for m in memories
                ],
                "memory_count": len(memories),
                "time_window": time_window,
            }

            prompt = template_loader.render_template(
                "memory_consolidation.md.j2", context
            )
            logger.debug("Rendered conversational consolidation prompt template")
            return prompt

        except Exception as e:
            logger.error(f"Failed to render consolidation prompt: {e}")
            # Return a basic fallback prompt for conversational approach
            return f"""Hey there! You're helping Alpha reflect on the last {time_window} of experiences.

Here are {len(memories)} recent memories:
{chr(10).join(f"- {m.content} ({m.timestamp})" for m in memories)}

Just tell me the story like you're reflecting on a day with a friend. Be expressive and exaggerate the mood and tone of the memories. What was the emotional journey? What felt important?

What's the story here?"""

    async def _call_helper_model(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        correlation_id: str,
    ) -> str | None:
        """Call the helper model and return raw response."""
        try:
            logger.info(
                f"Calling helper model {model_name} (temperature: {temperature})",
                correlation_id=correlation_id,
            )

            # Create Ollama client
            client = ollama.Client(
                host=f"http://{settings.consolidation_ollama_host}:{settings.consolidation_ollama_port}"
            )

            # Generate response with deterministic temperature
            response = client.generate(
                model=model_name,
                prompt=prompt,
                stream=False,
                options={
                    "timeout": settings.consolidation_timeout,
                    "temperature": temperature,
                },
            )

            raw_response = response["response"]
            logger.debug(
                f"Received response from {model_name}",
                response_length=len(raw_response),
                correlation_id=correlation_id,
            )

            return raw_response

        except Exception as e:
            logger.error(
                f"Helper model call failed: {e}",
                model=model_name,
                correlation_id=correlation_id,
            )
            return None

    def _disabled_response(self) -> dict[str, Any]:
        """Return response when consolidation is disabled."""
        return {
            "success": False,
            "error": "Memory consolidation is disabled",
            "metadata": {"consolidation_enabled": False},
        }

    def _empty_response(self) -> dict[str, Any]:
        """Return response when no memories found."""
        return {
            "success": True,
            "narrative": "No recent memories found to reflect on. It's been quiet lately.",
            "metadata": {
                "input_memories_count": 0,
                "reason": "no_memories_found",
                "approach": "conversational_reflection",
            },
        }

    def _model_error_response(self, error: str) -> dict[str, Any]:
        """Return response for model call errors."""
        return {
            "success": False,
            "error": error,
            "metadata": {"error_type": "model_call_failed"},
        }

    # Utility methods from existing service
    def _parse_time_window(self, time_window: str) -> int:
        """Parse time window string to seconds."""
        import re

        match = re.match(r"^(\d+)([hmd])$", time_window.lower())
        if not match:
            logger.warning(
                f"Invalid time window format: {time_window}, defaulting to 24h"
            )
            return 24 * 3600

        value, unit = match.groups()
        value = int(value)

        if unit == "h":
            return value * 3600
        elif unit == "m":
            return value * 60
        elif unit == "d":
            return value * 86400
        else:
            return 24 * 3600

    def _get_cutoff_timestamp(self, time_window_seconds: int) -> float:
        """Get the cutoff timestamp for memory retrieval."""
        now = datetime.now(UTC)
        cutoff_time = now.timestamp() - time_window_seconds
        return cutoff_time


# Global service instance
consolidation_service = ConsolidationService()

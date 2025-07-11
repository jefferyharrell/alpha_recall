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
from ..services.tokenizer import tokenizer
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

            # Determine model to use for context optimization
            model_to_use = model_name or settings.helper_model

            # Generate system prompt template to calculate token budget
            system_prompt_template = self._get_system_prompt_template(time_window)

            # Get memories optimized for the model's context window
            memories = await self._get_memories_for_context_budget(
                time_window, model_to_use, system_prompt_template
            )

            if not memories:
                logger.debug("No recent memories found", correlation_id=correlation_id)
                return self._empty_response()

            # Prepare system prompt and memories data for chat API
            memories_data = "\n".join(f"{m.timestamp}: {m.content}" for m in memories)

            narrative_response = await self._call_helper_model(
                prompt="",  # Not used with chat API
                model_name=model_to_use,
                temperature=temperature,
                correlation_id=correlation_id,
                system_prompt=system_prompt_template,
                memories_data=memories_data,
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

    def _get_system_prompt_template(self, time_window: str) -> str:
        """Get the system prompt template for token budget calculation.

        This returns just the instruction part of the prompt without memories,
        so we can calculate token budget before selecting memories.

        Args:
            time_window: Time window for context

        Returns:
            System prompt template string
        """
        try:
            template = template_loader.load_template("memory_consolidation.md.j2")

            # Render with empty memories list to get just the system prompt
            return template.render(
                memories=[], time_window=time_window, template_only=True
            )
        except Exception as e:
            logger.error(f"Failed to render system prompt template: {e}")
            # Template loading is critical - don't silently fall back
            raise RuntimeError(f"System prompt template loading failed: {e}") from e

    async def _call_helper_model(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        correlation_id: str,
        system_prompt: str = None,
        memories_data: str = None,
    ) -> str | None:
        """Call the helper model using chat API with system/user message separation."""
        try:
            logger.info(
                f"Calling helper model {model_name} (temperature: {temperature})",
                correlation_id=correlation_id,
            )

            # Create Ollama client
            client = ollama.Client(
                host=f"http://{settings.consolidation_ollama_host}:{settings.consolidation_ollama_port}"
            )

            # Prepare options with dynamic context window
            context_window = self._get_model_context_window(model_name)
            options = {
                "timeout": settings.consolidation_timeout,
                "num_ctx": context_window,
            }

            # Add temperature if specified (None = use model default)
            if temperature is not None:
                options["temperature"] = temperature

            # Use chat API with system/user separation if provided
            if system_prompt and memories_data:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": memories_data},
                ]

                response = client.chat(
                    model=model_name,
                    messages=messages,
                    stream=False,
                    options=options,
                )

                return response.get("message", {}).get("content", "").strip()
            else:
                # Fallback to generate API for backward compatibility
                response = client.generate(
                    model=model_name,
                    prompt=prompt,
                    stream=False,
                    options=options,
                )

                raw_response = response["response"]
                logger.debug(
                    f"Received response from {model_name}",
                    response_length=len(raw_response),
                    correlation_id=correlation_id,
                )
                return raw_response.strip()

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

    # Dynamic context optimization methods
    def _get_model_context_window(self, model_name: str) -> int:
        """Get the context window size for the specified model.

        Args:
            model_name: Name of the model to check

        Returns:
            Context window size in tokens, defaults to 4096 if unknown
        """
        try:
            # Use same client configuration as helper model calls
            client = ollama.Client(
                host=f"http://{settings.consolidation_ollama_host}:{settings.consolidation_ollama_port}"
            )

            model_info = client.show(model_name)
            # Check for num_ctx parameter in model details
            if "details" in model_info and "parameters" in model_info["details"]:
                params = model_info["details"]["parameters"]
                if "num_ctx" in params:
                    context_window = int(params["num_ctx"])
                    logger.debug(
                        f"Retrieved context window from Ollama: {context_window} for {model_name}"
                    )
                    return context_window

            # Default context window for common models
            model_defaults = {
                "llama3.2:1b": 2048,
                "llama3.2:3b": 4096,
                "llama3.1:8b": 32768,
                "granite3.3:2b": 8192,
                "granite3.3:8b": 8192,
            }

            default_context = model_defaults.get(model_name, 4096)
            logger.debug(
                f"Using default context window: {default_context} for {model_name}"
            )
            return default_context

        except Exception as e:
            logger.error(f"Failed to get context window for {model_name}: {e}")
            # Fall back to defaults if Ollama is unreachable
            model_defaults = {
                "llama3.2:1b": 2048,
                "llama3.2:3b": 4096,
                "llama3.1:8b": 32768,
                "granite3.3:2b": 8192,
                "granite3.3:8b": 8192,
            }
            fallback_context = model_defaults.get(model_name, 4096)
            logger.warning(
                f"Using fallback context window: {fallback_context} for {model_name}"
            )
            return fallback_context

    async def _get_memories_for_context_budget(
        self,
        time_window: str,
        model_name: str,
        system_prompt: str,
    ) -> list[ShortTermMemory]:
        """Get memories optimized for the model's context window.

        Args:
            time_window: Maximum time window to consider
            model_name: Model name to optimize for
            system_prompt: System prompt to account for in token budget

        Returns:
            List of memories that fit within the context budget
        """
        # Get model context window
        max_context = self._get_model_context_window(model_name)

        # Calculate token budget
        system_tokens = tokenizer.count(system_prompt)
        response_buffer = 2000  # Reserve tokens for response
        safety_margin = 200  # Safety buffer

        available_tokens = max_context - system_tokens - response_buffer - safety_margin

        logger.debug(
            "Token budget calculation",
            max_context=max_context,
            system_tokens=system_tokens,
            response_buffer=response_buffer,
            safety_margin=safety_margin,
            available_tokens=available_tokens,
        )

        # Get all memories within time window (chronologically)
        all_memories = await self._get_recent_memories_for_consolidation(time_window)

        # Select memories that fit within token budget
        selected_memories = []
        token_budget = available_tokens

        for memory in all_memories:  # Already sorted newest first
            memory_tokens = tokenizer.count(memory.content)

            if token_budget >= memory_tokens:
                selected_memories.append(memory)
                token_budget -= memory_tokens
            else:
                break  # Stop when we'd overflow context

        logger.info(
            "Dynamic memory selection completed",
            total_available=len(all_memories),
            selected_count=len(selected_memories),
            tokens_used=available_tokens - token_budget,
            tokens_remaining=token_budget,
        )

        return selected_memories

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

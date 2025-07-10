"""Memory consolidation service with schema validation and systematic model evaluation."""

import json
import time
from datetime import UTC, datetime
from typing import Any

import ollama
from pydantic import ValidationError

from ..config import settings
from ..logging import get_logger
from ..schemas.consolidation import (
    ConsolidationInput,
    ConsolidationOutput,
    ConsolidationValidationError,
    ShortTermMemory,
)
from ..services.redis import get_redis_service
from ..services.template_loader import template_loader
from ..utils.correlation import generate_correlation_id, set_correlation_id

logger = get_logger(__name__)


class ConsolidationService:
    """Memory consolidation service with systematic model evaluation capabilities."""

    def __init__(self):
        """Initialize the consolidation service."""
        self.redis_service = get_redis_service()

    async def consolidate_shortterm_memories(
        self,
        time_window: str = "24h",
        model_name: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Consolidate short-term memories with clean schema validation.

        Args:
            time_window: Time window for memory retrieval (e.g., "24h", "7d", "30m")
            model_name: Override default helper model for testing
            temperature: Model temperature (default 0.0 for deterministic testing)

        Returns:
            Dictionary containing consolidation results or failure information
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

            # Validate input schema
            try:
                consolidation_input = ConsolidationInput(
                    memories=memories,
                    time_window=time_window,
                    consolidation_prompt=self._get_prompt_template(memories),
                )
            except ValidationError as e:
                logger.error(
                    "Input validation failed",
                    error=str(e),
                    correlation_id=correlation_id,
                )
                return self._validation_error_response("input", str(e))

            # Call helper model for consolidation
            model_to_use = model_name or settings.helper_model
            raw_response = await self._call_helper_model(
                consolidation_input.consolidation_prompt,
                model_to_use,
                temperature,
                correlation_id,
            )

            if raw_response is None:
                return self._model_error_response("Helper model call failed")

            # Attempt to validate output schema
            validation_result = self._validate_model_output(
                raw_response, correlation_id
            )

            processing_time_ms = int((time.time() - start_time) * 1000)

            if validation_result["success"]:
                # Successfully validated output
                consolidation_output = validation_result["parsed_output"]
                consolidation_output.consolidation_metadata.update(
                    {
                        "processing_time_ms": processing_time_ms,
                        "model_used": model_to_use,
                        "temperature": temperature,
                        "input_memories_count": len(memories),
                        "time_window": time_window,
                        "validation_success": True,
                    }
                )

                logger.info(
                    "Memory consolidation completed successfully",
                    model=model_to_use,
                    processing_time_ms=processing_time_ms,
                    entities_count=len(consolidation_output.entities),
                    insights_count=len(consolidation_output.insights),
                    correlation_id=correlation_id,
                )

                return {
                    "success": True,
                    "consolidation": consolidation_output.dict(),
                    "metadata": {
                        "processing_time_ms": processing_time_ms,
                        "model_evaluation": {
                            "model_name": model_to_use,
                            "temperature": temperature,
                            "validation_success": True,
                            "structural_correctness": "valid",
                        },
                    },
                    "correlation_id": correlation_id,
                }

            else:
                # Validation failed - return debug information
                logger.warning(
                    "Model output validation failed",
                    model=model_to_use,
                    validation_errors=validation_result["errors"],
                    correlation_id=correlation_id,
                )

                return {
                    "success": False,
                    "error": "Model output validation failed",
                    "validation_errors": validation_result["errors"],
                    "raw_model_output": raw_response,
                    "metadata": {
                        "processing_time_ms": processing_time_ms,
                        "model_evaluation": {
                            "model_name": model_to_use,
                            "temperature": temperature,
                            "validation_success": False,
                            "structural_correctness": "invalid",
                            "failure_patterns": validation_result["errors"],
                        },
                        "input_memories_count": len(memories),
                        "time_window": time_window,
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
                    "model_evaluation": {
                        "model_name": model_name or settings.helper_model,
                        "validation_success": False,
                        "structural_correctness": "error",
                    },
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

    def _get_prompt_template(self, memories: list[ShortTermMemory]) -> str:
        """Render the consolidation prompt template."""
        try:
            context = {
                "memories": [
                    {"content": m.content, "timestamp": m.timestamp} for m in memories
                ],
                "memory_count": len(memories),
                "model": settings.helper_model,
                "schema_requirements": self._get_schema_requirements_text(),
            }

            prompt = template_loader.render_template(
                "memory_consolidation.md.j2", context
            )
            logger.debug("Rendered consolidation prompt template")
            return prompt

        except Exception as e:
            logger.error(f"Failed to render consolidation prompt: {e}")
            # Return a basic fallback prompt with schema requirements
            return f"""Please consolidate these {len(memories)} memories into structured JSON matching this schema:

{self._get_schema_requirements_text()}

Memories:
{chr(10).join(f"- {m.content}" for m in memories)}

Return only valid JSON."""

    def _get_schema_requirements_text(self) -> str:
        """Get human-readable schema requirements for the prompt."""
        return """
Required JSON structure:
{
  "entities": [{"name": "string", "entity_type": "string", "description": "string"}],
  "relationships": [{"from_entity": "string", "to_entity": "string", "relationship_type": "string"}],
  "insights": [{"insight": "string", "category": "string", "importance": "low|medium|high|critical"}],
  "summary": "string",
  "emotional_context": "string",
  "next_steps": ["string"]
}

All fields are required, but arrays can be empty if no relevant information is found.
        """.strip()

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

    def _validate_model_output(
        self, raw_response: str, correlation_id: str
    ) -> dict[str, Any]:
        """Validate model output against schema and return detailed results."""
        try:
            # First, try to extract and parse JSON
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1

            if json_start == -1 or json_end <= json_start:
                return {
                    "success": False,
                    "errors": [
                        ConsolidationValidationError(
                            field="json_structure",
                            error="No JSON found in response",
                            expected_format="Valid JSON object with { and }",
                        ).dict()
                    ],
                }

            json_text = raw_response[json_start:json_end]

            try:
                parsed_json = json.loads(json_text)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "errors": [
                        ConsolidationValidationError(
                            field="json_parsing",
                            error=f"JSON decode error: {str(e)}",
                            expected_format="Valid JSON syntax",
                        ).dict()
                    ],
                }

            # Now validate against Pydantic schema
            try:
                consolidation_output = ConsolidationOutput(**parsed_json)
                logger.debug(
                    "Model output validation successful", correlation_id=correlation_id
                )
                return {"success": True, "parsed_output": consolidation_output}

            except ValidationError as e:
                validation_errors = []
                for error in e.errors():
                    validation_errors.append(
                        ConsolidationValidationError(
                            field=".".join(str(loc) for loc in error["loc"]),
                            error=error["msg"],
                            expected_format=f"Type: {error.get('type', 'unknown')}",
                        ).dict()
                    )

                return {"success": False, "errors": validation_errors}

        except Exception as e:
            return {
                "success": False,
                "errors": [
                    ConsolidationValidationError(
                        field="validation_process",
                        error=f"Validation process failed: {str(e)}",
                        expected_format="Valid consolidation output schema",
                    ).dict()
                ],
            }

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
            "consolidation": ConsolidationOutput().dict(),
            "metadata": {"input_memories_count": 0, "reason": "no_memories_found"},
        }

    def _validation_error_response(self, stage: str, error: str) -> dict[str, Any]:
        """Return response for validation errors."""
        return {
            "success": False,
            "error": f"Validation failed at {stage}: {error}",
            "metadata": {"validation_stage": stage},
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

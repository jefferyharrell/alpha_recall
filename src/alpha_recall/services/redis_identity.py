"""Redis service for identity facts management."""

import time
from typing import Any

from ..logging import get_logger
from ..services.time import time_service
from ..utils.correlation import (
    create_child_correlation_id,
    get_correlation_id,
)
from .redis_base import RedisBaseService

logger = get_logger("services.redis_identity")


class RedisIdentityService(RedisBaseService):
    """Service for managing identity facts in Redis."""

    def add_identity_fact(self, fact: str, score: float = None) -> dict[str, Any]:
        """Add or update an identity fact in the ordered set.

        Args:
            fact: The identity fact to add
            score: Optional score for ordering (defaults to auto-append)

        Returns:
            Dict with success status, fact, score, position, and metadata
        """
        correlation_id = get_correlation_id() or create_child_correlation_id(
            "add_identity_fact"
        )
        start_time = time.perf_counter()

        try:
            identity_key = "identity_facts"

            # Check if fact already exists
            existing_score = self.client.zscore(identity_key, fact)
            is_update = existing_score is not None

            # Auto-assign score if not provided
            if score is None:
                if is_update:
                    score = existing_score  # Keep existing score
                else:
                    # Get highest score and add 1.0
                    highest = self.client.zrevrange(identity_key, 0, 0, withscores=True)
                    if highest:
                        score = highest[0][1] + 1.0
                    else:
                        score = 1.0

            # Add/update the fact
            self.client.zadd(identity_key, {fact: score})

            # Get position (1-indexed)
            position = self.client.zrank(identity_key, fact) + 1

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Identity fact added/updated",
                fact_length=len(fact),
                score=score,
                position=position,
                is_update=is_update,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "success": True,
                "fact": fact,
                "score": score,
                "position": position,
                "created_at": time_service.now()["iso_datetime"],
                "updated": is_update,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Error adding identity fact",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return {
                "success": False,
                "error": f"Failed to add identity fact: {e}",
            }

    def update_identity_fact(self, fact: str, new_score: float) -> dict[str, Any]:
        """Update the score/position of an existing identity fact.

        Args:
            fact: The identity fact to update
            new_score: New score for positioning

        Returns:
            Dict with success status, fact, old/new scores and positions
        """
        correlation_id = get_correlation_id() or create_child_correlation_id(
            "update_identity_fact"
        )
        start_time = time.perf_counter()

        try:
            identity_key = "identity_facts"

            # Check if fact exists
            old_score = self.client.zscore(identity_key, fact)
            if old_score is None:
                # Get available facts for error message
                all_facts = [
                    f.decode("utf-8") for f in self.client.zrange(identity_key, 0, -1)
                ]
                return {
                    "success": False,
                    "error": f"Identity fact not found: {fact}",
                    "available_facts": all_facts[:10],  # Limit for readability
                }

            # Get old position
            old_position = self.client.zrank(identity_key, fact) + 1

            # Update the score
            self.client.zadd(identity_key, {fact: new_score})

            # Get new position
            new_position = self.client.zrank(identity_key, fact) + 1

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Identity fact updated",
                fact_length=len(fact),
                old_score=old_score,
                new_score=new_score,
                old_position=old_position,
                new_position=new_position,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "success": True,
                "fact": fact,
                "old_score": old_score,
                "new_score": new_score,
                "old_position": old_position,
                "new_position": new_position,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Error updating identity fact",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return {
                "success": False,
                "error": f"Failed to update identity fact: {e}",
            }

    def get_identity_facts(self) -> list[dict[str, Any]]:
        """Get all identity facts in order.

        Returns:
            List of identity facts with scores and positions
        """
        correlation_id = get_correlation_id() or create_child_correlation_id(
            "get_identity_facts"
        )
        start_time = time.perf_counter()

        try:
            identity_key = "identity_facts"

            # Get all facts with scores
            facts_with_scores = self.client.zrange(identity_key, 0, -1, withscores=True)

            result = []
            for i, (fact_bytes, score) in enumerate(facts_with_scores):
                fact = fact_bytes.decode("utf-8")
                result.append(
                    {
                        "content": fact,
                        "score": score,
                        "position": i + 1,
                    }
                )

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.debug(
                "Identity facts retrieved",
                count=len(result),
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return result

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Error retrieving identity facts",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return []

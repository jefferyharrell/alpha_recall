"""Redis service for short-term memory operations."""

import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import redis

from ..config import settings
from ..logging import get_logger
from ..services.time import time_service
from ..utils.correlation import (
    create_child_correlation_id,
    get_correlation_id,
    set_correlation_id,
)

logger = get_logger("services.redis")


class RedisService:
    """Service for interacting with Redis database."""

    def __init__(self):
        """Initialize the Redis service."""
        self._client: redis.Redis | None = None
        self._connection_tested = False

    @property
    def client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(settings.redis_uri)
            logger.debug("Created Redis client", uri=settings.redis_uri)
        return self._client

    def test_connection(self) -> bool:
        """Test the Redis connection."""
        if self._connection_tested:
            return True

        try:
            start_time = time.perf_counter()
            # Simple ping test
            result = self.client.ping()
            test_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            if result:
                self._connection_tested = True
                logger.info(
                    "Redis connection test successful",
                    test_time_ms=test_time_ms,
                    correlation_id=get_correlation_id(),
                )
                return True
            else:
                logger.error(
                    "Redis connection test failed - ping returned False",
                    correlation_id=get_correlation_id(),
                )
                return False
        except Exception as e:
            logger.error(
                "Redis connection test failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            return False

    def store_memory(
        self,
        memory_id: str,
        content: str,
        semantic_embedding: np.ndarray,
        emotional_embedding: np.ndarray,
        created_at: str,
    ) -> bool:
        """Store a memory with its embeddings to Redis.

        Args:
            memory_id: Unique identifier for the memory
            content: The memory content
            semantic_embedding: The semantic embedding vector
            emotional_embedding: The emotional embedding vector
            created_at: ISO timestamp when memory was created

        Returns:
            True if stored successfully, False otherwise
        """
        store_corr_id = create_child_correlation_id("redis_store")
        set_correlation_id(store_corr_id)
        start_time = time.perf_counter()

        try:
            # Store memory content and metadata as a hash
            memory_key = f"memory:{memory_id}"

            # Convert embeddings to Python lists for storage
            if isinstance(semantic_embedding, np.ndarray):
                semantic_list = semantic_embedding.tolist()
            else:
                semantic_list = semantic_embedding

            if isinstance(emotional_embedding, np.ndarray):
                emotional_list = emotional_embedding.tolist()
            else:
                emotional_list = emotional_embedding

            # Store semantic vector as binary for Redis vector search
            semantic_vector_binary = np.array(semantic_list, dtype=np.float32).tobytes()

            self.client.hset(
                memory_key,
                mapping={
                    "content": content,
                    "created_at": created_at,
                    "id": memory_id,
                    "semantic_vector": semantic_vector_binary,  # Binary format for vector search
                    "emotional_vector": json.dumps(emotional_list),
                },
            )

            # Set TTL on the memory
            self.client.expire(memory_key, settings.redis_ttl)

            # Add to chronological index for browsing
            timestamp = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            ).timestamp()
            self.client.zadd("memory_index", {memory_id: timestamp})
            self.client.expire("memory_index", settings.redis_ttl)

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Memory stored to Redis successfully",
                memory_id=memory_id,
                content_length=len(content),
                semantic_dims=len(semantic_embedding),
                emotional_dims=len(emotional_embedding),
                operation_time_ms=operation_time_ms,
                correlation_id=store_corr_id,
            )

            return True

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to store memory to Redis",
                memory_id=memory_id,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=store_corr_id,
            )
            return False

    def ensure_vector_index_exists(self) -> bool:
        """Ensure Redis vector search index exists for memories.

        Returns:
            True if index exists or was created successfully, False otherwise
        """
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Check if index already exists
            self.client.execute_command("FT.INFO", "memory_semantic_index")
            logger.debug(
                "Vector index already exists",
                correlation_id=correlation_id,
            )
            return True
        except Exception as e:
            # Handle both redis.ResponseError and other Redis exceptions
            error_str = str(e)
            logger.debug(
                f"Index check exception: {repr(error_str)}",
                correlation_id=correlation_id,
            )
            if "Unknown index name" in error_str or "no such index" in error_str:
                # Index doesn't exist, create it
                logger.info(
                    "Creating vector search index",
                    correlation_id=correlation_id,
                )

                try:
                    # Create index for memory hashes with semantic vector field
                    self.client.execute_command(
                        "FT.CREATE",
                        "memory_semantic_index",
                        "ON",
                        "HASH",
                        "PREFIX",
                        "1",
                        "memory:",
                        "SCHEMA",
                        "content",
                        "TEXT",
                        "created_at",
                        "TEXT",
                        "id",
                        "TEXT",
                        "semantic_vector",
                        "VECTOR",
                        "FLAT",
                        "6",
                        "TYPE",
                        "FLOAT32",
                        "DIM",
                        "768",
                        "DISTANCE_METRIC",
                        "COSINE",
                    )

                    operation_time_ms = round(
                        (time.perf_counter() - start_time) * 1000, 2
                    )
                    logger.info(
                        "Vector search index created successfully",
                        operation_time_ms=operation_time_ms,
                        correlation_id=correlation_id,
                    )
                    return True
                except Exception as create_error:
                    operation_time_ms = round(
                        (time.perf_counter() - start_time) * 1000, 2
                    )
                    logger.error(
                        "Failed to create vector index",
                        error=str(create_error),
                        error_type=type(create_error).__name__,
                        operation_time_ms=operation_time_ms,
                        correlation_id=correlation_id,
                    )
                    return False
            else:
                logger.error(
                    "Unexpected error checking vector index",
                    error=str(e),
                    error_type=type(e).__name__,
                    correlation_id=correlation_id,
                )
                return False

    def search_related_memories(
        self,
        content: str,
        query_embedding: np.ndarray,
        exclude_id: str = None,
    ) -> list[dict[str, Any]]:
        """Search for related memories using Redis vector search (splash functionality).

        This function implements the 'splash' - finding related memories that come up
        when a new memory is added to the system using Redis's native vector search.

        Args:
            content: The new memory content
            query_embedding: The semantic embedding of the new content
            exclude_id: Memory ID to exclude from results (e.g., the memory we just stored)

        Returns:
            List of related memories with similarity scores
        """
        search_corr_id = create_child_correlation_id("splash_search")
        set_correlation_id(search_corr_id)
        start_time = time.perf_counter()

        try:
            # First check if we have any memories to search
            memory_count = self.client.zcard("memory_index")
            if memory_count == 0:
                logger.info(
                    "No memories found in index",
                    correlation_id=search_corr_id,
                )
                return []

            # Ensure vector index exists
            if not self.ensure_vector_index_exists():
                logger.error(
                    "Vector index unavailable and could not be created",
                    correlation_id=search_corr_id,
                )
                return []

            # Convert embedding to binary format for Redis vector search
            if isinstance(query_embedding, np.ndarray):
                vector_bytes = query_embedding.astype(np.float32).tobytes()
            else:
                vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

            # Build the vector search query
            # Search for top 6 results (we'll filter out exclude_id if needed)
            search_limit = 6 if exclude_id else 5

            search_query = f"*=>[KNN {search_limit} @semantic_vector $query_vector AS similarity_score]"

            try:
                # Execute vector search
                search_result = self.client.execute_command(
                    "FT.SEARCH",
                    "memory_semantic_index",
                    search_query,
                    "PARAMS",
                    "2",
                    "query_vector",
                    vector_bytes,
                    "SORTBY",
                    "similarity_score",
                    "ASC",  # Lower scores = more similar in cosine distance
                    "RETURN",
                    "4",
                    "content",
                    "created_at",
                    "id",
                    "similarity_score",
                    "DIALECT",
                    "2",
                )

                # Parse search results
                total_results = search_result[0]
                logger.info(
                    f"Vector search returned {total_results} results",
                    correlation_id=search_corr_id,
                )

                related_memories = []

                # Results come in pairs: [doc_key, [field1, value1, field2, value2, ...]]
                for i in range(1, len(search_result), 2):
                    doc_key = search_result[i].decode("utf-8")
                    doc_fields = search_result[i + 1]

                    # Extract memory ID from document key (format: "memory:id")
                    memory_id = doc_key.replace("memory:", "")

                    # Skip if this is the excluded memory
                    if exclude_id and memory_id == exclude_id:
                        continue

                    # Parse field values (they come as [field, value, field, value, ...])
                    fields = {}
                    for j in range(0, len(doc_fields), 2):
                        field_name = doc_fields[j].decode("utf-8")
                        field_value = doc_fields[j + 1].decode("utf-8")
                        fields[field_name] = field_value

                    # Convert similarity score from distance to similarity (1 - distance)
                    distance_score = float(fields.get("similarity_score", 1.0))
                    similarity_score = 1.0 - distance_score

                    # Only include memories with reasonable similarity (> 0.3)
                    if similarity_score > 0.3:
                        related_memories.append(
                            {
                                "content": fields.get("content", ""),
                                "similarity_score": similarity_score,
                                "created_at": fields.get("created_at", ""),
                                "id": fields.get("id", memory_id),
                                "source": "redis_vector_search",
                            }
                        )

                # Results are already sorted by similarity from Redis
                # Limit to top 5 results
                related_memories = related_memories[:5]

                operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

                logger.info(
                    "Vector search completed",
                    related_found=len(related_memories),
                    top_similarity=(
                        related_memories[0]["similarity_score"]
                        if related_memories
                        else 0.0
                    ),
                    operation_time_ms=operation_time_ms,
                    correlation_id=search_corr_id,
                )

                return related_memories

            except Exception as search_error:
                operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.error(
                    "Vector search failed",
                    error=str(search_error),
                    error_type=type(search_error).__name__,
                    operation_time_ms=operation_time_ms,
                    correlation_id=search_corr_id,
                )
                return []

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Splash search failed",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=search_corr_id,
            )
            return []

    def browse_memories(
        self,
        limit: int = 20,
        offset: int = 0,
        since: str | None = None,
        search: str | None = None,
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Browse short-term memories with filtering and pagination.

        Args:
            limit: Maximum number of memories to return
            offset: Number of memories to skip for pagination
            since: Time duration to look back (e.g., "6h", "2d", "1w")
            search: Optional text to filter memory content
            order: Sort order - "desc" for newest first, "asc" for oldest first

        Returns:
            List of memories with metadata
        """
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Get memory IDs from the chronological index
            if order == "desc":
                # Newest first (descending by timestamp)
                memory_ids = self.client.zrevrange(
                    "memory_index", offset, offset + limit - 1
                )
            else:
                # Oldest first (ascending by timestamp)
                memory_ids = self.client.zrange(
                    "memory_index", offset, offset + limit - 1
                )

            memories = []
            for memory_id in memory_ids:
                memory_id_str = memory_id.decode("utf-8")
                memory_key = f"memory:{memory_id_str}"

                # Get memory data
                memory_data = self.client.hgetall(memory_key)
                if not memory_data:
                    continue

                # Decode and parse memory data
                memory = {}
                for key, value in memory_data.items():
                    key_str = key.decode("utf-8")
                    if key_str == "emotional_vector":
                        # Skip emotional vector for browsing
                        continue
                    elif key_str == "semantic_vector":
                        # Skip semantic vector for browsing
                        continue
                    else:
                        memory[key_str] = value.decode("utf-8")

                # Apply search filter if provided
                if search and search.lower() not in memory.get("content", "").lower():
                    continue

                # Apply time filter if provided
                if since:
                    # This would need pendulum parsing logic for time filtering
                    # For now, we'll skip this filter
                    pass

                memories.append(memory)

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Memory browsing completed",
                memories_found=len(memories),
                limit=limit,
                offset=offset,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return memories

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Memory browsing failed",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return []

    def search_memories(
        self,
        query: str,
        query_embedding: np.ndarray,
        search_type: str = "semantic",
        limit: int = 10,
        through_the_last: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories using semantic or emotional similarity.

        Args:
            query: The search query text
            query_embedding: The embedding of the search query
            search_type: "semantic" or "emotional" search type
            limit: Maximum number of memories to return
            through_the_last: Time duration to look back

        Returns:
            List of memories with similarity scores
        """
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            if search_type == "semantic":
                # Use vector search for semantic search
                return self.search_related_memories(query, query_embedding)
            else:
                # For emotional search, fall back to text search for now
                # This would need emotional vector search implementation
                memories = self.browse_memories(
                    limit=limit * 2
                )  # Get more for filtering

                # Simple text matching for emotional search fallback
                filtered_memories = []
                for memory in memories:
                    if query.lower() in memory.get("content", "").lower():
                        filtered_memories.append(
                            {
                                **memory,
                                "similarity_score": 0.8,  # Mock similarity score
                                "search_type": "text_matching",
                            }
                        )

                operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

                logger.info(
                    "Emotional search completed (text fallback)",
                    memories_found=len(filtered_memories),
                    query=query,
                    operation_time_ms=operation_time_ms,
                    correlation_id=correlation_id,
                )

                return filtered_memories[:limit]

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Memory search failed",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return []

    # Identity Facts Management

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


# Global service instance
_redis_service: RedisService | None = None


def get_redis_service() -> RedisService:
    """Get the global Redis service instance."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service

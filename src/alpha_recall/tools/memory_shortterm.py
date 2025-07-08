"""Short-term memory tools for Alpha-Recall v1.0.0."""

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pendulum
import redis
from fastmcp import FastMCP

from ..config import settings
from ..logging import get_logger
from ..services.embedding import embedding_service
from ..utils.correlation import create_child_correlation_id, set_correlation_id

# Explicitly declare public interface
__all__ = ["remember_shortterm", "browse_shortterm", "register_shortterm_memory_tools"]


def get_redis_client() -> redis.Redis:
    """Get a Redis client instance."""
    logger = get_logger("tools.redis_client")
    logger.info(
        "Creating Redis client",
        redis_uri=settings.redis_uri,
        operation="redis_client_create",
    )
    return redis.from_url(settings.redis_uri)


def store_memory_to_redis(
    client: redis.Redis,
    memory_id: str,
    content: str,
    semantic_embedding: np.ndarray,
    emotional_embedding: np.ndarray,
    created_at: str,
) -> bool:
    """Store a memory with its embeddings to Redis.

    Args:
        client: Redis client instance
        memory_id: Unique identifier for the memory
        content: The memory content
        semantic_embedding: The semantic embedding vector
        emotional_embedding: The emotional embedding vector
        created_at: ISO timestamp when memory was created

    Returns:
        True if stored successfully, False otherwise
    """
    logger = get_logger("tools.redis_store")
    store_corr_id = create_child_correlation_id("redis_store")
    set_correlation_id(store_corr_id)

    try:
        # Store memory content and metadata as a hash
        memory_key = f"memory:{memory_id}"

        # Store basic fields and vectors
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

        client.hset(
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
        client.expire(memory_key, settings.redis_ttl)

        # Add to chronological index for browsing
        timestamp = datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        ).timestamp()
        client.zadd("memory_index", {memory_id: timestamp})
        client.expire("memory_index", settings.redis_ttl)

        logger.info(
            "Memory stored to Redis successfully",
            memory_id=memory_id,
            content_length=len(content),
            semantic_dims=len(semantic_embedding),
            emotional_dims=len(emotional_embedding),
            operation="redis_store",
        )

        return True

    except Exception as e:
        logger.error(
            "Failed to store memory to Redis",
            memory_id=memory_id,
            error=str(e),
            operation="redis_store",
        )
        return False


def ensure_vector_index_exists(client: redis.Redis) -> bool:
    """Ensure Redis vector search index exists for memories.

    Returns:
        True if index exists or was created successfully, False otherwise
    """
    logger = get_logger("tools.vector_index")

    try:
        # Check if index already exists
        client.execute_command("FT.INFO", "memory_semantic_index")
        logger.debug("Vector index already exists", operation="vector_index_check")
        return True
    except Exception as e:
        # Handle both redis.ResponseError and other Redis exceptions
        error_str = str(e)
        logger.debug(
            f"Index check exception: {repr(error_str)}", operation="vector_index_check"
        )
        if "Unknown index name" in error_str or "no such index" in error_str:
            # Index doesn't exist, create it
            logger.info("Creating vector search index", operation="vector_index_create")

            try:
                # Create index for memory hashes with semantic vector field
                client.execute_command(
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
                logger.info(
                    "Vector search index created successfully",
                    operation="vector_index_create",
                )
                return True
            except Exception as create_error:
                logger.error(
                    "Failed to create vector index",
                    error=str(create_error),
                    operation="vector_index_create",
                )
                return False
        else:
            logger.error(
                "Unexpected error checking vector index",
                error=str(e),
                operation="vector_index_check",
            )
            return False


def search_related_memories(
    client: redis.Redis,
    content: str,
    query_embedding: np.ndarray,
    exclude_id: str = None,
) -> list[dict[str, Any]]:
    """Search for related memories using Redis vector search (splash functionality).

    This function implements the 'splash' - finding related memories that come up
    when a new memory is added to the system using Redis's native vector search.

    Args:
        client: Redis client instance
        content: The new memory content
        query_embedding: The semantic embedding of the new content
        exclude_id: Memory ID to exclude from results (e.g., the memory we just stored)

    Returns:
        List of related memories with similarity scores
    """
    logger = get_logger("tools.splash_search")
    search_corr_id = create_child_correlation_id("splash_search")
    set_correlation_id(search_corr_id)

    try:
        # First check if we have any memories to search
        memory_count = client.zcard("memory_index")
        if memory_count == 0:
            logger.info("No memories found in index", operation="splash_search")
            return []

        # Ensure vector index exists
        if not ensure_vector_index_exists(client):
            logger.error(
                "Vector index unavailable and could not be created",
                operation="splash_search",
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
            search_result = client.execute_command(
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
                operation="splash_search",
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

            logger.info(
                "Vector search completed",
                related_found=len(related_memories),
                top_similarity=(
                    related_memories[0]["similarity_score"] if related_memories else 0.0
                ),
                operation="splash_search",
            )

            return related_memories

        except Exception as search_error:
            logger.error(
                "Vector search failed",
                error=str(search_error),
                operation="splash_search",
            )
            return []

    except Exception as e:
        logger.error("Splash search failed", error=str(e), operation="splash_search")
        return []


def remember_shortterm(content: str) -> str:
    """Store a short-term memory with semantic and emotional embeddings.

    This function processes memory content, generates semantic and emotional embeddings,
    stores the memory in Redis, and returns related memories via the 'splash' functionality.

    Args:
        content: The memory content to store

    Returns:
        JSON string with storage confirmation, performance metrics, and related memories
    """
    # Create correlation ID for this memory operation
    memory_corr_id = create_child_correlation_id("remember_shortterm")
    set_correlation_id(memory_corr_id)

    start_time = time.perf_counter()
    logger = get_logger("tools.remember_shortterm")

    logger.info(
        "Processing short-term memory",
        content_length=len(content),
        content_preview=content[:100] + "..." if len(content) > 100 else content,
        operation="remember_shortterm",
    )

    # Generate semantic embedding
    semantic_start = time.perf_counter()
    semantic_embedding = embedding_service.encode_semantic(content)
    semantic_time = time.perf_counter() - semantic_start

    # Generate emotional embedding
    emotional_start = time.perf_counter()
    emotional_embedding = embedding_service.encode_emotional(content)
    emotional_time = time.perf_counter() - emotional_start

    # Generate unique memory ID and timestamp
    memory_id = f"stm_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
    created_at = datetime.now(UTC).isoformat()

    # Create single Redis client for this tool call
    client = get_redis_client()

    # Store memory with embeddings to Redis
    store_start = time.perf_counter()
    store_success = store_memory_to_redis(
        client=client,
        memory_id=memory_id,
        content=content,
        semantic_embedding=semantic_embedding,
        emotional_embedding=emotional_embedding,
        created_at=created_at,
    )
    store_time = time.perf_counter() - store_start

    if not store_success:
        logger.warning(
            "Failed to store memory to Redis",
            memory_id=memory_id,
            operation="remember_shortterm",
        )

    total_time = time.perf_counter() - start_time

    # Calculate performance metrics
    content_tokens = len(content.split()) * 1.3  # Rough token estimate
    semantic_tps = content_tokens / semantic_time if semantic_time > 0 else 0
    emotional_tps = content_tokens / emotional_time if emotional_time > 0 else 0

    result = {
        "status": "stored" if store_success else "processed_only",
        "memory_id": memory_id,
        "created_at": created_at,
        "content_length": len(content),
        "content_tokens": int(content_tokens),
        "semantic_embedding_dims": len(semantic_embedding),
        "emotional_embedding_dims": len(emotional_embedding),
        "timing": {
            "total_ms": round(total_time * 1000, 2),
            "semantic_ms": round(semantic_time * 1000, 2),
            "emotional_ms": round(emotional_time * 1000, 2),
            "storage_ms": round(store_time * 1000, 2),
        },
        "performance": {
            "semantic_tokens_per_sec": round(semantic_tps, 1),
            "emotional_tokens_per_sec": round(emotional_tps, 1),
            "total_tokens_per_sec": round(content_tokens / total_time, 1),
        },
        "storage": {"success": store_success, "backend": "redis"},
        "correlation_id": memory_corr_id,
    }

    logger.info(
        "Short-term memory processed successfully",
        content_tokens=int(content_tokens),
        semantic_dims=len(semantic_embedding),
        emotional_dims=len(emotional_embedding),
        total_time_ms=round(total_time * 1000, 2),
        semantic_tps=round(semantic_tps, 1),
        emotional_tps=round(emotional_tps, 1),
        operation="remember_shortterm",
    )

    # SPLASH: Search for related memories using cosine similarity (exclude the one we just stored)
    splash_start = time.perf_counter()
    related_memories = search_related_memories(
        client,
        content,
        semantic_embedding,
        exclude_id=memory_id if store_success else None,
    )
    splash_time = time.perf_counter() - splash_start

    # Add splash metrics to result
    result.update(
        {
            "splash": {
                "related_memories_found": len(related_memories),
                "search_time_ms": round(splash_time * 1000, 2),
                "memories": related_memories[:5],  # Top 5 most related
            }
        }
    )

    logger.info(
        "Memory splash completed",
        related_count=len(related_memories),
        splash_time_ms=round(splash_time * 1000, 2),
        operation="remember_shortterm_splash",
    )

    # Clean up embedding arrays from memory
    del semantic_embedding, emotional_embedding

    return json.dumps(result, indent=2)


def browse_shortterm(
    limit: int = 20,
    offset: int = 0,
    since: str | None = None,
    search: str | None = None,
    order: str = "desc",
) -> str:
    """Browse short-term memories with filtering and pagination.

    Args:
        limit: Maximum number of memories to return (default 20)
        offset: Number of memories to skip for pagination (default 0)
        since: Time duration to look back (e.g., "6h", "2d", "1w") using Pendulum syntax
        search: Optional text to filter memory content (case-insensitive)
        order: Sort order - "desc" for newest first, "asc" for oldest first

    Returns:
        JSON string with paginated memories and metadata
    """
    # Create correlation ID for this browse operation
    browse_corr_id = create_child_correlation_id("browse_shortterm")
    set_correlation_id(browse_corr_id)

    start_time = time.perf_counter()
    logger = get_logger("tools.browse_shortterm")

    logger.info(
        "Browsing short-term memories",
        limit=limit,
        offset=offset,
        since=since,
        search_query=search,
        order=order,
        operation="browse_shortterm",
    )

    # Create Redis client
    client = get_redis_client()

    try:
        # Calculate time range if 'since' is provided
        if since:
            try:
                # Try parsing as a duration string first (e.g., "6h", "2d", "1w")
                now = pendulum.now()

                # Handle common duration formats
                if since.endswith("h"):
                    hours = int(since[:-1])
                    cutoff_timestamp = now.subtract(hours=hours).timestamp()
                elif since.endswith("d"):
                    days = int(since[:-1])
                    cutoff_timestamp = now.subtract(days=days).timestamp()
                elif since.endswith("w"):
                    weeks = int(since[:-1])
                    cutoff_timestamp = now.subtract(weeks=weeks).timestamp()
                elif since.endswith("m"):
                    minutes = int(since[:-1])
                    cutoff_timestamp = now.subtract(minutes=minutes).timestamp()
                else:
                    # Try parsing as absolute datetime with Pendulum
                    parsed_time = pendulum.parse(since)
                    cutoff_timestamp = parsed_time.timestamp()

            except Exception as e:
                logger.warning(
                    "Invalid since parameter, ignoring",
                    since=since,
                    error=str(e),
                    operation="browse_shortterm",
                )
                cutoff_timestamp = 0
        else:
            cutoff_timestamp = 0

        # Get memory IDs from chronological index within time range
        if order == "desc":
            # Newest first - use ZREVRANGEBYSCORE
            if cutoff_timestamp > 0:
                memory_ids_with_scores = client.zrevrangebyscore(
                    "memory_index",
                    "+inf",
                    cutoff_timestamp,
                    start=offset,
                    num=limit,
                    withscores=True,
                )
                total_in_range = client.zcount("memory_index", cutoff_timestamp, "+inf")
            else:
                memory_ids_with_scores = client.zrevrange(
                    "memory_index", offset, offset + limit - 1, withscores=True
                )
                total_in_range = client.zcard("memory_index")
        else:
            # Oldest first - use ZRANGEBYSCORE
            if cutoff_timestamp > 0:
                memory_ids_with_scores = client.zrangebyscore(
                    "memory_index",
                    cutoff_timestamp,
                    "+inf",
                    start=offset,
                    num=limit,
                    withscores=True,
                )
                total_in_range = client.zcount("memory_index", cutoff_timestamp, "+inf")
            else:
                memory_ids_with_scores = client.zrange(
                    "memory_index", offset, offset + limit - 1, withscores=True
                )
                total_in_range = client.zcard("memory_index")

        logger.info(
            "Retrieved memory IDs from index",
            found_count=len(memory_ids_with_scores),
            total_in_range=total_in_range,
            operation="browse_shortterm",
        )

        memories = []
        now = pendulum.now()

        # Fetch memory content for each ID
        for memory_id_bytes, timestamp in memory_ids_with_scores:
            memory_id = memory_id_bytes.decode("utf-8")
            memory_key = f"memory:{memory_id}"

            # Get memory data from hash
            memory_data = client.hmget(memory_key, ["content", "created_at", "id"])

            if memory_data[0] is not None:  # Content exists
                content = memory_data[0].decode("utf-8")
                created_at = memory_data[1].decode("utf-8") if memory_data[1] else ""
                stored_id = (
                    memory_data[2].decode("utf-8") if memory_data[2] else memory_id
                )

                # Apply search filter if provided
                if search and search.lower() not in content.lower():
                    continue

                # Calculate human-readable age
                try:
                    created_time = pendulum.parse(created_at)
                    age = created_time.diff_for_humans(now)
                except Exception:
                    age = "unknown"

                memories.append(
                    {
                        "id": stored_id,
                        "content": content,
                        "created_at": created_at,
                        "age": age,
                        "timestamp": timestamp,
                    }
                )

        # If we applied search filtering, we might have fewer results
        actual_returned = len(memories)
        has_more = (offset + actual_returned) < total_in_range

        total_time = time.perf_counter() - start_time

        result = {
            "memories": memories,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "returned": actual_returned,
                "total_in_range": total_in_range,
                "has_more": has_more,
                "showing": (
                    f"{offset + 1}-{offset + actual_returned} of {total_in_range}"
                    if actual_returned > 0
                    else "0 of 0"
                ),
            },
            "filters": {
                "since": since,
                "search": search,
                "order": order,
            },
            "timing": {
                "total_ms": round(total_time * 1000, 2),
            },
            "correlation_id": browse_corr_id,
        }

        logger.info(
            "Browse short-term memories completed",
            returned=actual_returned,
            total_in_range=total_in_range,
            total_time_ms=round(total_time * 1000, 2),
            operation="browse_shortterm",
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(
            "Failed to browse short-term memories",
            error=str(e),
            operation="browse_shortterm",
        )
        return json.dumps(
            {
                "error": f"Failed to browse memories: {str(e)}",
                "memories": [],
                "pagination": {"returned": 0, "total_in_range": 0},
                "correlation_id": browse_corr_id,
            },
            indent=2,
        )


def register_shortterm_memory_tools(mcp: FastMCP) -> None:
    """Register short-term memory tools with the MCP server."""
    logger = get_logger("tools.memory_shortterm")

    # Register the short-term memory tools
    mcp.tool(remember_shortterm)
    mcp.tool(browse_shortterm)

    logger.debug("Short-term memory tools registered")

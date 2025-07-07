"""Short-term memory tools for Alpha-Recall v1.0.0."""

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import redis
from fastmcp import FastMCP

from ..config import settings
from ..logging import get_logger
from ..services.embedding import embedding_service
from ..utils.correlation import create_child_correlation_id, set_correlation_id

# Explicitly declare public interface
__all__ = ["remember_shortterm", "register_shortterm_memory_tools"]


def get_redis_client() -> redis.Redis:
    """Get a Redis client instance."""
    return redis.from_url(settings.redis_uri)


def store_memory_to_redis(
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
    logger = get_logger("tools.redis_store")
    store_corr_id = create_child_correlation_id("redis_store")
    set_correlation_id(store_corr_id)

    try:
        client = get_redis_client()

        # Store memory content and metadata as a hash
        memory_key = f"memory:{memory_id}"

        # Store basic fields
        client.hset(
            memory_key,
            mapping={
                "content": content,
                "created_at": created_at,
                "id": memory_id,
                "semantic_vector": json.dumps(semantic_embedding.tolist()),
                "emotional_vector": json.dumps(emotional_embedding.tolist()),
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


def search_related_memories(
    content: str, query_embedding: np.ndarray, exclude_id: str = None
) -> list[dict[str, Any]]:
    """Search for related memories using cosine similarity (splash functionality).

    This function implements the 'splash' - finding related memories that come up
    when a new memory is added to the system.

    Args:
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
        client = get_redis_client()

        # Get all memory IDs from the chronological index (most recent first)
        memory_ids = client.zrevrange("memory_index", 0, -1)

        if not memory_ids:
            logger.info("No memories found in index", operation="splash_search")
            return []

        related_memories = []
        checked_count = 0

        # Check each memory for similarity
        for memory_id_bytes in memory_ids:
            memory_id = memory_id_bytes.decode("utf-8")

            if exclude_id and memory_id == exclude_id:
                continue

            checked_count += 1

            # Get memory data
            memory_data = client.hgetall(f"memory:{memory_id}")

            if not memory_data:
                continue

            # Decode bytes to strings
            memory_data = {
                k.decode("utf-8"): v.decode("utf-8") for k, v in memory_data.items()
            }

            # Get semantic embedding for similarity calculation
            semantic_vector_json = memory_data.get("semantic_vector")

            if not semantic_vector_json:
                continue

            # Parse vector data
            stored_embedding = np.array(json.loads(semantic_vector_json))

            # Calculate cosine similarity
            # Normalize vectors for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            stored_norm = stored_embedding / np.linalg.norm(stored_embedding)

            # Cosine similarity
            similarity = float(np.dot(query_norm, stored_norm))

            # Only include memories with reasonable similarity (> 0.3)
            if similarity > 0.3:
                related_memories.append(
                    {
                        "content": memory_data.get("content", ""),
                        "similarity_score": similarity,
                        "created_at": memory_data.get("created_at", ""),
                        "id": memory_data.get("id", memory_id),
                        "source": "redis_search",
                    }
                )

        # Sort by similarity score (highest first)
        related_memories.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Limit to top 5 results
        related_memories = related_memories[:5]

        logger.info(
            "Splash search completed",
            related_found=len(related_memories),
            top_similarity=(
                related_memories[0]["similarity_score"] if related_memories else 0.0
            ),
            checked_memories=checked_count,
            operation="splash_search",
        )

        return related_memories

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

    # Store memory with embeddings to Redis
    store_start = time.perf_counter()
    store_success = store_memory_to_redis(
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
        content, semantic_embedding, exclude_id=memory_id if store_success else None
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


def register_shortterm_memory_tools(mcp: FastMCP) -> None:
    """Register short-term memory tools with the MCP server."""
    logger = get_logger("tools.memory_shortterm")

    # Register the remember_shortterm tool
    mcp.tool(remember_shortterm)

    logger.debug("Short-term memory tools registered")

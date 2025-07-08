"""Remember shortterm tool for Alpha-Recall v1.0.0."""

import json
import time
import uuid
from datetime import UTC, datetime

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.embedding import embedding_service
from ..utils.correlation import create_child_correlation_id, set_correlation_id
from .utils.redis_stm import (
    get_redis_client,
    search_related_memories,
    store_memory_to_redis,
)

__all__ = ["remember_shortterm", "register_remember_shortterm_tool"]


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


def register_remember_shortterm_tool(mcp: FastMCP) -> None:
    """Register the remember_shortterm tool with the MCP server."""
    logger = get_logger("tools.remember_shortterm")

    # Register the tool
    mcp.tool(remember_shortterm)

    logger.debug("remember_shortterm tool registered")

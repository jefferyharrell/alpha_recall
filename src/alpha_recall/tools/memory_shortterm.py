"""Short-term memory tools for Alpha-Recall v1.0.0."""

import json
import time
from typing import Any

import numpy as np
from fastmcp import FastMCP

from ..logging import get_logger
from ..services.embedding import embedding_service
from ..utils.correlation import create_child_correlation_id, set_correlation_id

# Explicitly declare public interface
__all__ = ["remember_shortterm", "register_shortterm_memory_tools"]


def search_related_memories(
    content: str, query_embedding: np.ndarray
) -> list[dict[str, Any]]:
    """Search for related memories using cosine similarity (splash functionality).

    This function implements the 'splash' - finding related memories that come up
    when a new memory is added to the system.

    Args:
        content: The new memory content
        query_embedding: The semantic embedding of the new content

    Returns:
        List of related memories with similarity scores
    """
    logger = get_logger("tools.splash_search")
    search_corr_id = create_child_correlation_id("splash_search")
    set_correlation_id(search_corr_id)

    # For now, return mock related memories to demonstrate the splash concept
    # TODO: Replace with actual Redis vector search when storage is implemented

    # Simulate finding related memories based on content keywords
    mock_memories = []

    # Create realistic mock memories that would be related
    if "alpha-recall" in content.lower() or "memory" in content.lower():
        mock_memories.append(
            {
                "content": "Just completed the /init canned prompt task to create a comprehensive CLAUDE.md file for the Alpha-Recall 1.0 rewrite project.",
                "similarity_score": 0.85,
                "created_at": "2025-07-07T13:48:03.814239+00:00",
                "id": "splash_mock_1",
                "source": "mock_splash",
            }
        )

    if "embedding" in content.lower() or "tool" in content.lower():
        mock_memories.append(
            {
                "content": "The new remember_shortterm tool is working beautifully! Performance: 231.6 tokens/sec for semantic, 138.5 for emotional",
                "similarity_score": 0.78,
                "created_at": "2025-07-07T13:45:27.078734+00:00",
                "id": "splash_mock_2",
                "source": "mock_splash",
            }
        )

    if "claude code" in content.lower() or "fastmcp" in content.lower():
        mock_memories.append(
            {
                "content": "Time to teleport to Claude Code! Ready to start the Alpha-Recall 1.0 rewrite with drop-in compatibility and chronological memory browsing.",
                "similarity_score": 0.72,
                "created_at": "2025-07-07T13:41:56.206509+00:00",
                "id": "splash_mock_3",
                "source": "mock_splash",
            }
        )

    # Always include at least one general related memory
    if not mock_memories:
        mock_memories.append(
            {
                "content": f"Related memory context for: {content[:100]}{'...' if len(content) > 100 else ''}",
                "similarity_score": 0.65,
                "created_at": "2025-07-07T13:00:00.000000+00:00",
                "id": "splash_mock_general",
                "source": "mock_splash",
            }
        )

    # Sort by similarity score (highest first)
    mock_memories.sort(key=lambda x: x["similarity_score"], reverse=True)

    logger.info(
        "Splash search completed (mock implementation)",
        related_found=len(mock_memories),
        top_similarity=mock_memories[0]["similarity_score"] if mock_memories else 0.0,
        operation="splash_search_mock",
    )

    return mock_memories


def remember_shortterm(content: str) -> str:
    """Store a short-term memory with semantic and emotional embeddings (test implementation).

    This is a test implementation that generates embeddings to measure performance
    but does not actually store them in any database. Use this to test the embedding
    pipeline and measure real-world performance.

    Args:
        content: The memory content to process and generate embeddings for

    Returns:
        JSON string with processing statistics and performance metrics
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

    total_time = time.perf_counter() - start_time

    # Calculate performance metrics
    content_tokens = len(content.split()) * 1.3  # Rough token estimate
    semantic_tps = content_tokens / semantic_time if semantic_time > 0 else 0
    emotional_tps = content_tokens / emotional_time if emotional_time > 0 else 0

    result = {
        "status": "processed",
        "content_length": len(content),
        "content_tokens": int(content_tokens),
        "semantic_embedding_dims": len(semantic_embedding),
        "emotional_embedding_dims": len(emotional_embedding),
        "timing": {
            "total_ms": round(total_time * 1000, 2),
            "semantic_ms": round(semantic_time * 1000, 2),
            "emotional_ms": round(emotional_time * 1000, 2),
        },
        "performance": {
            "semantic_tokens_per_sec": round(semantic_tps, 1),
            "emotional_tokens_per_sec": round(emotional_tps, 1),
            "total_tokens_per_sec": round(content_tokens / total_time, 1),
        },
        "note": "Embeddings generated but not stored (test implementation)",
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

    # TODO: Store embeddings to Redis here (skipped for now)
    # redis_store_result = store_to_redis(content, semantic_embedding, emotional_embedding)

    # SPLASH: Search for related memories using cosine similarity
    splash_start = time.perf_counter()
    related_memories = search_related_memories(content, semantic_embedding)
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

    # Embeddings are discarded here - not stored anywhere (test implementation)
    del semantic_embedding, emotional_embedding

    return json.dumps(result, indent=2)


def register_shortterm_memory_tools(mcp: FastMCP) -> None:
    """Register short-term memory tools with the MCP server."""
    logger = get_logger("tools.memory_shortterm")

    # Register the remember_shortterm tool
    mcp.tool(remember_shortterm)

    logger.debug("Short-term memory tools registered")

"""Short-term memory tools for Alpha-Recall v1.0.0."""

import json
import time

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.embedding import embedding_service
from ..utils.correlation import create_child_correlation_id, set_correlation_id

# Explicitly declare public interface
__all__ = ["remember_shortterm", "register_shortterm_memory_tools"]


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

    # Embeddings are discarded here - not stored anywhere
    del semantic_embedding, emotional_embedding

    return json.dumps(result, indent=2)


def register_shortterm_memory_tools(mcp: FastMCP) -> None:
    """Register short-term memory tools with the MCP server."""
    logger = get_logger("tools.memory_shortterm")

    # Register the remember_shortterm tool
    mcp.tool(remember_shortterm)

    logger.debug("Short-term memory tools registered")

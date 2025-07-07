"""Demo memory tool to showcase correlation ID flow."""

import time

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.embedding import embedding_service
from ..utils.correlation import (
    generate_correlation_id,
    set_correlation_id,
)

__all__ = ["demo_remember", "register_memory_demo_tools"]

logger = get_logger("tools.memory_demo")


def demo_remember(content: str) -> str:
    """Demo tool showing correlation ID flow through memory operations."""
    # Generate correlation ID for this memory operation
    correlation_id = generate_correlation_id("mem")
    set_correlation_id(correlation_id)

    operation_start = time.perf_counter()

    logger.info(
        "Memory operation started",
        operation="demo_remember",
        content_length=len(content),
        content_preview=content[:50] + "..." if len(content) > 50 else content,
    )

    # Step 1: Generate semantic embedding (with child correlation ID)
    semantic_start = time.perf_counter()
    logger.debug(
        "Starting semantic embedding generation",
        step="semantic_embedding",
        operation="demo_remember",
    )

    semantic_embedding = embedding_service.encode_semantic(content)
    semantic_time_ms = (time.perf_counter() - semantic_start) * 1000

    logger.debug(
        "Semantic embedding completed",
        step="semantic_embedding",
        operation="demo_remember",
        embedding_dimensions=len(semantic_embedding),
        generation_time_ms=round(semantic_time_ms, 2),
    )

    # Step 2: Generate emotional embedding (with child correlation ID)
    emotional_start = time.perf_counter()
    logger.debug(
        "Starting emotional embedding generation",
        step="emotional_embedding",
        operation="demo_remember",
    )

    emotional_embedding = embedding_service.encode_emotional(content)
    emotional_time_ms = (time.perf_counter() - emotional_start) * 1000

    logger.debug(
        "Emotional embedding completed",
        step="emotional_embedding",
        operation="demo_remember",
        embedding_dimensions=len(emotional_embedding),
        generation_time_ms=round(emotional_time_ms, 2),
    )

    # Step 3: Simulate memory storage
    storage_start = time.perf_counter()
    logger.debug(
        "Starting memory storage simulation",
        step="memory_storage",
        operation="demo_remember",
        semantic_dims=len(semantic_embedding),
        emotional_dims=len(emotional_embedding),
    )

    # Simulate storage delay
    time.sleep(0.01)  # 10ms delay

    storage_time_ms = (time.perf_counter() - storage_start) * 1000
    memory_id = f"mem_{correlation_id.split('_')[1]}"

    logger.debug(
        "Memory storage completed",
        step="memory_storage",
        operation="demo_remember",
        memory_id=memory_id,
        storage_time_ms=round(storage_time_ms, 2),
    )

    # Step 4: Simulate related memory search
    search_start = time.perf_counter()
    logger.debug(
        "Starting related memory search",
        step="related_search",
        operation="demo_remember",
        search_vector_dims=len(semantic_embedding),
    )

    # Simulate search delay and results
    time.sleep(0.005)  # 5ms delay
    related_count = 3  # Simulated results

    search_time_ms = (time.perf_counter() - search_start) * 1000

    logger.debug(
        "Related memory search completed",
        step="related_search",
        operation="demo_remember",
        related_memories_found=related_count,
        search_time_ms=round(search_time_ms, 2),
    )

    total_time_ms = (time.perf_counter() - operation_start) * 1000

    # Final summary log
    logger.info(
        "Memory operation completed successfully",
        operation="demo_remember",
        memory_id=memory_id,
        total_time_ms=round(total_time_ms, 2),
        breakdown={
            "semantic_embedding_ms": round(semantic_time_ms, 2),
            "emotional_embedding_ms": round(emotional_time_ms, 2),
            "storage_ms": round(storage_time_ms, 2),
            "search_ms": round(search_time_ms, 2),
        },
        content_characters=len(content),
        related_memories_found=related_count,
        embedding_dimensions={
            "semantic": len(semantic_embedding),
            "emotional": len(emotional_embedding),
        },
    )

    return f"Memory stored successfully with ID: {memory_id}"


def register_memory_demo_tools(mcp: FastMCP) -> None:
    """Register memory demo tools with the MCP server."""
    logger.debug("Registering memory demo tools")

    # Register the demo tool
    mcp.tool(demo_remember)

    logger.debug("Memory demo tools registered")

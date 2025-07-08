"""Search shortterm tool for Alpha-Recall v1.0.0."""

import json
import time
from typing import Any

import pendulum
from fastmcp import FastMCP

from ..logging import get_logger
from ..services.embedding import embedding_service
from ..utils.correlation import create_child_correlation_id, set_correlation_id
from .utils.redis_stm import ensure_vector_index_exists, get_redis_client

__all__ = ["search_shortterm", "register_search_shortterm_tool"]


def search_shortterm(
    query: str,
    limit: int = 10,
    search_type: str = "semantic",
    through_the_last: str | None = None,
) -> str:
    """Search short-term memories using semantic or emotional similarity.

    Args:
        query: The search query text
        limit: Maximum number of memories to return (default 10)
        search_type: "semantic" or "emotional" search type (default "semantic")
        through_the_last: Time duration to look back (e.g., "6h", "2d", "1w") using Pendulum syntax

    Returns:
        JSON string with search results, timing, and metadata
    """
    # Create correlation ID for this search operation
    search_corr_id = create_child_correlation_id("search_shortterm")
    set_correlation_id(search_corr_id)

    start_time = time.perf_counter()
    logger = get_logger("tools.search_shortterm")

    logger.info(
        "Searching short-term memories",
        query_length=len(query),
        query_preview=query[:100] + "..." if len(query) > 100 else query,
        limit=limit,
        search_type=search_type,
        through_the_last=through_the_last,
        operation="search_shortterm",
    )

    # Validate search_type
    if search_type not in ["semantic", "emotional"]:
        error_msg = (
            f"Invalid search_type '{search_type}'. Must be 'semantic' or 'emotional'"
        )
        logger.error(error_msg, operation="search_shortterm")
        return json.dumps(
            {
                "error": error_msg,
                "memories": [],
                "timing": {"total_ms": 0},
                "correlation_id": search_corr_id,
            },
            indent=2,
        )

    # Create Redis client
    client = get_redis_client()

    try:
        # Calculate time range if 'through_the_last' is provided
        if through_the_last:
            try:
                # Parse duration string (e.g., "6h", "2d", "1w")
                now = pendulum.now()

                if through_the_last.endswith("h"):
                    hours = int(through_the_last[:-1])
                    cutoff_timestamp = now.subtract(hours=hours).timestamp()
                elif through_the_last.endswith("d"):
                    days = int(through_the_last[:-1])
                    cutoff_timestamp = now.subtract(days=days).timestamp()
                elif through_the_last.endswith("w"):
                    weeks = int(through_the_last[:-1])
                    cutoff_timestamp = now.subtract(weeks=weeks).timestamp()
                elif through_the_last.endswith("m"):
                    minutes = int(through_the_last[:-1])
                    cutoff_timestamp = now.subtract(minutes=minutes).timestamp()
                else:
                    # Try parsing as absolute datetime with Pendulum
                    parsed_time = pendulum.parse(through_the_last)
                    cutoff_timestamp = parsed_time.timestamp()

            except Exception as e:
                logger.warning(
                    "Invalid through_the_last parameter, ignoring",
                    through_the_last=through_the_last,
                    error=str(e),
                    operation="search_shortterm",
                )
                cutoff_timestamp = 0
        else:
            cutoff_timestamp = 0

        # Check if we have any memories to search
        memory_count = client.zcard("memory_index")
        if memory_count == 0:
            logger.info("No memories found in index", operation="search_shortterm")
            return json.dumps(
                {
                    "memories": [],
                    "search_metadata": {
                        "query": query,
                        "search_type": search_type,
                        "limit": limit,
                        "through_the_last": through_the_last,
                        "total_memories_available": 0,
                    },
                    "timing": {
                        "total_ms": round((time.perf_counter() - start_time) * 1000, 2)
                    },
                    "correlation_id": search_corr_id,
                },
                indent=2,
            )

        # For semantic search, use Redis vector search
        if search_type == "semantic":
            memories = _search_semantic(client, query, limit, cutoff_timestamp, logger)
        else:
            # For emotional search, fall back to text matching for now
            # TODO: Implement proper emotional vector search when emotional embeddings are stored
            memories = _search_emotional_fallback(
                client, query, limit, cutoff_timestamp, logger
            )

        total_time = time.perf_counter() - start_time

        result = {
            "memories": memories,
            "search_metadata": {
                "query": query,
                "search_type": search_type,
                "limit": limit,
                "through_the_last": through_the_last,
                "results_found": len(memories),
                "total_memories_available": memory_count,
            },
            "timing": {
                "total_ms": round(total_time * 1000, 2),
            },
            "correlation_id": search_corr_id,
        }

        logger.info(
            "Search short-term memories completed",
            results_found=len(memories),
            search_type=search_type,
            total_time_ms=round(total_time * 1000, 2),
            operation="search_shortterm",
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(
            "Failed to search short-term memories",
            error=str(e),
            operation="search_shortterm",
        )
        return json.dumps(
            {
                "error": f"Search failed: {str(e)}",
                "memories": [],
                "search_metadata": {
                    "query": query,
                    "search_type": search_type,
                    "limit": limit,
                },
                "timing": {
                    "total_ms": round((time.perf_counter() - start_time) * 1000, 2)
                },
                "correlation_id": search_corr_id,
            },
            indent=2,
        )


def _search_semantic(
    client, query: str, limit: int, cutoff_timestamp: float, logger
) -> list[dict[str, Any]]:
    """Perform semantic vector search using Redis."""
    # Generate semantic embedding for the query
    embedding_start = time.perf_counter()
    query_embedding = embedding_service.encode_semantic(query)
    embedding_time = time.perf_counter() - embedding_start

    logger.info(
        "Query embedding generated",
        embedding_dims=len(query_embedding),
        embedding_time_ms=round(embedding_time * 1000, 2),
        operation="search_shortterm_semantic",
    )

    # Ensure vector index exists
    if not ensure_vector_index_exists(client):
        logger.error(
            "Vector index unavailable and could not be created",
            operation="search_shortterm_semantic",
        )
        return []

    # Convert embedding to binary format for Redis vector search
    import numpy as np

    if isinstance(query_embedding, np.ndarray):
        vector_bytes = query_embedding.astype(np.float32).tobytes()
    else:
        vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

    # Build the vector search query
    search_query = (
        f"*=>[KNN {limit} @semantic_vector $query_vector AS similarity_score]"
    )

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
            operation="search_shortterm_semantic",
        )

        memories = []
        now = pendulum.now()

        # Results come in pairs: [doc_key, [field1, value1, field2, value2, ...]]
        for i in range(1, len(search_result), 2):
            doc_key = search_result[i].decode("utf-8")
            doc_fields = search_result[i + 1]

            # Extract memory ID from document key (format: "memory:id")
            memory_id = doc_key.replace("memory:", "")

            # Parse field values (they come as [field, value, field, value, ...])
            fields = {}
            for j in range(0, len(doc_fields), 2):
                field_name = doc_fields[j].decode("utf-8")
                field_value = doc_fields[j + 1].decode("utf-8")
                fields[field_name] = field_value

            # Check time filter if specified
            if cutoff_timestamp > 0:
                try:
                    created_at = fields.get("created_at", "")
                    memory_timestamp = pendulum.parse(created_at).timestamp()
                    if memory_timestamp < cutoff_timestamp:
                        continue
                except Exception:
                    # Skip if we can't parse the timestamp
                    continue

            # Convert similarity score from distance to similarity (1 - distance)
            distance_score = float(fields.get("similarity_score", 1.0))
            similarity_score = 1.0 - distance_score

            # Calculate human-readable age
            try:
                created_time = pendulum.parse(fields.get("created_at", ""))
                age = created_time.diff_for_humans(now)
            except Exception:
                age = "unknown"

            memories.append(
                {
                    "id": fields.get("id", memory_id),
                    "content": fields.get("content", ""),
                    "created_at": fields.get("created_at", ""),
                    "age": age,
                    "similarity_score": similarity_score,
                    "search_type": "semantic",
                }
            )

        return memories[:limit]  # Ensure we don't exceed the requested limit

    except Exception as search_error:
        logger.error(
            "Semantic vector search failed",
            error=str(search_error),
            operation="search_shortterm_semantic",
        )
        return []


def _search_emotional_fallback(
    client, query: str, limit: int, cutoff_timestamp: float, logger
) -> list[dict[str, Any]]:
    """Fallback emotional search using text matching."""
    logger.info(
        "Using text-based fallback for emotional search",
        operation="search_shortterm_emotional",
    )

    # Get all memory IDs within time range
    if cutoff_timestamp > 0:
        memory_ids_with_scores = client.zrangebyscore(
            "memory_index", cutoff_timestamp, "+inf", withscores=True
        )
    else:
        memory_ids_with_scores = client.zrevrange(
            "memory_index", 0, -1, withscores=True
        )

    memories = []
    now = pendulum.now()
    query_lower = query.lower()

    # Search through memories for text matches
    for memory_id_bytes, _timestamp in memory_ids_with_scores:
        memory_id = memory_id_bytes.decode("utf-8")
        memory_key = f"memory:{memory_id}"

        # Get memory data from hash
        memory_data = client.hmget(memory_key, ["content", "created_at", "id"])

        if memory_data[0] is not None:  # Content exists
            content = memory_data[0].decode("utf-8")
            created_at = memory_data[1].decode("utf-8") if memory_data[1] else ""
            stored_id = memory_data[2].decode("utf-8") if memory_data[2] else memory_id

            # Simple text matching (case-insensitive)
            if query_lower in content.lower():
                # Calculate human-readable age
                try:
                    created_time = pendulum.parse(created_at)
                    age = created_time.diff_for_humans(now)
                except Exception:
                    age = "unknown"

                # Simple relevance score based on query frequency in content
                content_lower = content.lower()
                relevance_score = content_lower.count(query_lower) / len(
                    content_lower.split()
                )

                memories.append(
                    {
                        "id": stored_id,
                        "content": content,
                        "created_at": created_at,
                        "age": age,
                        "relevance_score": relevance_score,
                        "search_type": "emotional_fallback",
                    }
                )

        # Stop if we have enough results
        if len(memories) >= limit:
            break

    # Sort by relevance score (descending)
    memories.sort(key=lambda m: m.get("relevance_score", 0), reverse=True)
    return memories[:limit]


def register_search_shortterm_tool(mcp: FastMCP) -> None:
    """Register the search_shortterm tool with the MCP server."""
    logger = get_logger("tools.search_shortterm")

    # Register the tool
    mcp.tool(search_shortterm)

    logger.debug("search_shortterm tool registered")

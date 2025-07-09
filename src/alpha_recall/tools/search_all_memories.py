"""Search all memories tool for unified cross-system memory search."""

import json
import time

from fastmcp import FastMCP

from ..logging import get_logger
from ..utils.correlation import generate_correlation_id, set_correlation_id
from .get_entity import get_entity
from .search_longterm import search_longterm
from .search_narratives import search_narratives
from .search_shortterm import search_shortterm

__all__ = ["search_all_memories", "register_search_all_memories_tools"]


async def search_all_memories(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """Search across all memory systems (STM, LTM, NM) with unified results.

    This is a unified memory search tool that searches across all three memory subsystems:
    - Short-term memories (Redis with TTL) - both semantic and emotional search
    - Long-term observations (Memgraph) - semantic search only
    - Narrative memories (Redis with embeddings) - both semantic and emotional search
    - Entity names (exact matching)

    Results are merged and sorted by similarity score to provide
    serendipitous memory discovery across all storage systems.

    Args:
        query: Search query to find relevant memories
        limit: Maximum number of results to return (default: 10)
        offset: Number of results to skip for pagination (default: 0)

    Returns:
        JSON string containing unified search results from all memory systems
    """
    correlation_id = generate_correlation_id("search_all_memories")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.search_all_memories")
    logger.info(
        "Search all memories request received",
        query=query,
        limit=limit,
        offset=offset,
        correlation_id=correlation_id,
    )

    start_time = time.time()

    try:
        all_results = []

        # Search STM (short-term memory) - both semantic and emotional
        try:
            stm_semantic_result = search_shortterm(
                query, limit=50, search_type="semantic"
            )
            stm_semantic_data = json.loads(stm_semantic_result)

            if "memories" in stm_semantic_data:
                for memory in stm_semantic_data["memories"]:
                    all_results.append(
                        {
                            "source": "STM",
                            "search_type": "semantic",
                            "content": memory.get("content", ""),
                            "score": memory.get("similarity_score", 0.0),
                            "created_at": memory.get("created_at"),
                            "age": memory.get("age"),
                            "id": f"stm_semantic_{memory.get('id', hash(memory.get('content', '')))}",
                            "memory_id": memory.get("id"),
                        }
                    )
        except Exception as e:
            logger.warning(f"STM semantic search failed: {e}")

        try:
            stm_emotional_result = search_shortterm(
                query, limit=50, search_type="emotional"
            )
            stm_emotional_data = json.loads(stm_emotional_result)

            if "memories" in stm_emotional_data:
                for memory in stm_emotional_data["memories"]:
                    all_results.append(
                        {
                            "source": "STM",
                            "search_type": "emotional",
                            "content": memory.get("content", ""),
                            "score": memory.get(
                                "similarity_score", memory.get("relevance_score", 0.0)
                            ),
                            "created_at": memory.get("created_at"),
                            "age": memory.get("age"),
                            "id": f"stm_emotional_{memory.get('id', hash(memory.get('content', '')))}",
                            "memory_id": memory.get("id"),
                        }
                    )
        except Exception as e:
            logger.warning(f"STM emotional search failed: {e}")

        # Search LTM (long-term memory observations)
        try:
            ltm_result = search_longterm(query, limit=50)
            ltm_data = json.loads(ltm_result)

            if "observations" in ltm_data:
                for obs in ltm_data["observations"]:
                    all_results.append(
                        {
                            "source": "LTM",
                            "search_type": "semantic",
                            "content": obs.get("observation", obs.get("content", "")),
                            "score": obs.get("similarity_score", obs.get("score", 0.0)),
                            "created_at": obs.get("created_at"),
                            "entity_name": obs.get("entity_name"),
                            "id": f"ltm_{obs.get('entity_name', '')}_{hash(obs.get('observation', obs.get('content', '')))}",
                            "observation_id": obs.get("id"),
                        }
                    )
        except Exception as e:
            logger.warning(f"LTM search failed: {e}")

        # Search NM (narrative memory) - both semantic and emotional
        try:
            narrative_semantic_result = await search_narratives(
                query, search_type="semantic", granularity="both", limit=25
            )
            narrative_semantic_data = json.loads(narrative_semantic_result)

            if "results" in narrative_semantic_data:
                for result in narrative_semantic_data["results"]:
                    all_results.append(
                        {
                            "source": "NM",
                            "search_type": "semantic",
                            "content": result.get("content", result.get("title", "")),
                            "score": result.get(
                                "similarity_score", result.get("score", 0.0)
                            ),
                            "created_at": result.get("created_at"),
                            "story_id": result.get("story_id"),
                            "title": result.get("title"),
                            "participants": result.get("participants", []),
                            "granularity": result.get("granularity"),
                            "id": f"nm_semantic_{result.get('story_id', hash(result.get('content', '')))}",
                        }
                    )
        except Exception as e:
            logger.warning(f"NM semantic search failed: {e}")

        try:
            narrative_emotional_result = await search_narratives(
                query, search_type="emotional", granularity="both", limit=25
            )
            narrative_emotional_data = json.loads(narrative_emotional_result)

            if "results" in narrative_emotional_data:
                for result in narrative_emotional_data["results"]:
                    all_results.append(
                        {
                            "source": "NM",
                            "search_type": "emotional",
                            "content": result.get("content", result.get("title", "")),
                            "score": result.get(
                                "similarity_score", result.get("score", 0.0)
                            ),
                            "created_at": result.get("created_at"),
                            "story_id": result.get("story_id"),
                            "title": result.get("title"),
                            "participants": result.get("participants", []),
                            "granularity": result.get("granularity"),
                            "id": f"nm_emotional_{result.get('story_id', hash(result.get('content', '')))}",
                        }
                    )
        except Exception as e:
            logger.warning(f"NM emotional search failed: {e}")

        # Search entities (exact name matching)
        try:
            entity_result = get_entity(query)
            entity_data = json.loads(entity_result)

            if entity_data.get("success") and "entity" in entity_data:
                entity = entity_data["entity"]
                all_results.append(
                    {
                        "source": "ENTITY",
                        "search_type": "exact_match",
                        "content": f"Entity: {query}",
                        "score": 1.0,  # Perfect match for exact entity name
                        "entity_name": query,
                        "entity_type": entity.get("entity_type"),
                        "created_at": entity.get("created_at"),
                        "updated_at": entity.get("updated_at"),
                        "observations_count": entity.get("observations_count", 0),
                        "id": f"entity_{hash(query)}",
                    }
                )
        except Exception as e:
            logger.warning(f"Entity search failed: {e}")

        # Sort all results by score (descending - higher scores are more relevant)
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Remove duplicates by ID (keeping highest scored version)
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = result.get("id")
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        # Apply pagination
        total_found = len(unique_results)
        paginated_results = unique_results[offset : offset + limit]

        search_time_ms = int((time.time() - start_time) * 1000)

        response = {
            "success": True,
            "search": {
                "query": query,
                "limit": limit,
                "offset": offset,
            },
            "results": paginated_results,
            "metadata": {
                "total_found": total_found,
                "returned_count": len(paginated_results),
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_found,
                "sources_searched": [
                    "STM_SEMANTIC",
                    "STM_EMOTIONAL",
                    "LTM",
                    "NM_SEMANTIC",
                    "NM_EMOTIONAL",
                    "ENTITIES",
                ],
                "search_time_ms": search_time_ms,
                "search_method": "unified_cross_system",
            },
            "correlation_id": correlation_id,
        }

        logger.info(
            "Search all memories completed",
            query=query,
            total_found=total_found,
            returned_count=len(paginated_results),
            search_time_ms=search_time_ms,
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(
            "Search all memories failed",
            query=query,
            error=str(e),
            error_type=type(e).__name__,
            correlation_id=correlation_id,
        )

        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "correlation_id": correlation_id,
        }

        return json.dumps(error_response, indent=2)


def register_search_all_memories_tools(mcp: FastMCP) -> None:
    """Register search_all_memories tools with the MCP server."""
    logger = get_logger("tools.search_all_memories")

    # Register the search_all_memories tool
    mcp.tool(search_all_memories)

    logger.debug("search_all_memories tools registered")

"""Gentle refresh tool for Alpha-Recall v2.0 - exact replica of v0.1.0 output format."""

import json

from fastmcp import FastMCP

from ..config import settings
from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..services.redis import get_redis_service
from ..services.time import time_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["gentle_refresh", "register_gentle_refresh_tools"]


async def gentle_refresh(query: str | None = None) -> str:
    """
    Simplified refresh function focused on temporal orientation rather than semantic search.

    Designed to solve memory orientation problems by providing:
    1. Current time information for temporal grounding
    2. Core identity observations (natural language facts, not relationship triples)
    3. 10 most recent short-term memories for contextual orientation
    4. Alpha-Snooze memory consolidation (if enabled) - processes recent memories for insights
    5. 5 most recent observations for slow-changing facts

    Eliminates cognitive overload from semantic search and prioritizes temporal
    orientation over semantic relevance. When Alpha-Snooze is enabled, provides
    additional memory consolidation insights extracted from recent interactions.

    Args:
        query: Optional query parameter (accepted for compatibility but ignored)

    Returns:
        JSON string containing:
        - time: Current time information
        - core_identity: Essential identity observations (observations only, no relationships)
        - shortterm_memories: 10 most recent short-term memories
        - memory_consolidation: Alpha-Snooze insights (if enabled and available)
        - recent_observations: 5 most recent observations
    """
    correlation_id = generate_correlation_id("gentle_refresh")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.gentle_refresh")
    logger.info("Gentle refresh tool called", correlation_id=correlation_id)

    try:
        # Initialize response structure
        response = {"success": True}

        # Get current time for temporal orientation
        response["time"] = await time_service.now_async()

        # Core identity: Get observations only (no relationship triples)
        core_identity_node = settings.core_identity_node
        logger.info(f"Loading core identity: {core_identity_node}")

        try:
            memgraph_service = get_memgraph_service()
            core_identity_entity = memgraph_service.get_entity_with_observations(
                core_identity_node
            )

            if core_identity_entity:
                # Only include observations, skip relationships to avoid analysis mode
                core_identity_filtered = {
                    "name": core_identity_entity.get("name", core_identity_node),
                    "updated_at": core_identity_entity.get("updated_at"),
                    "observations": core_identity_entity.get("observations", []),
                    # Deliberately exclude relationships to focus on identity facts
                }
                response["core_identity"] = core_identity_filtered
            else:
                logger.warning(f"Core identity node '{core_identity_node}' not found")
                response["core_identity"] = None

        except Exception as e:
            logger.error(f"Error loading core identity: {e}")
            response["core_identity"] = None

        # Short-term memories: Get 10 most recent for temporal orientation
        try:
            redis_service = get_redis_service()
            shortterm_limit = 10  # Increased from 5 for better temporal context
            logger.info(f"Retrieving {shortterm_limit} recent short-term memories")

            # Get recent memory IDs from the sorted set
            memory_ids_with_scores = redis_service.client.zrevrange(
                "memory_index", 0, shortterm_limit - 1, withscores=True
            )

            shortterm_memories = []
            for memory_id_bytes, _timestamp in memory_ids_with_scores:
                memory_id = memory_id_bytes.decode("utf-8")
                memory_key = f"memory:{memory_id}"

                # Get memory data from hash
                memory_data = redis_service.client.hmget(
                    memory_key, ["content", "created_at", "client_name"]
                )

                if memory_data[0] is not None:  # Content exists
                    content = memory_data[0].decode("utf-8")
                    created_at = (
                        memory_data[1].decode("utf-8") if memory_data[1] else ""
                    )
                    client_name = (
                        memory_data[2].decode("utf-8") if memory_data[2] else "unknown"
                    )

                    shortterm_memories.append(
                        {
                            "content": content,
                            "created_at": created_at,
                            "client": {"client_name": client_name},
                        }
                    )

            logger.info(f"Retrieved {len(shortterm_memories)} short-term memories")
            response["shortterm_memories"] = shortterm_memories

        except Exception as e:
            logger.error(f"Error retrieving short-term memories: {e}")
            response["shortterm_memories"] = []

        # TODO: Alpha-Snooze memory consolidation (placeholder for now)
        # This would integrate with the alpha-snooze system from v0.1.0
        response["memory_consolidation"] = {
            "entities": [],
            "relationships": [],
            "insights": [],
            "summary": "",
            "emotional_context": "",
            "next_steps": [],
            "processed_memories_count": 0,
            "consolidation_timestamp": time_service.utc_isoformat(),
            "model_used": "placeholder",
        }

        # Recent observations: Get 5 most recent for slow-changing facts
        try:
            memgraph_service = get_memgraph_service()
            recent_limit = 5
            logger.info(f"Retrieving {recent_limit} most recent observations")

            # Query for most recent observations across all entities
            query = """
            MATCH (e:Entity)-[:HAS_OBSERVATION]->(o:Observation)
            RETURN e.name as entity_name, o.content as content, o.created_at as created_at
            ORDER BY o.created_at DESC
            LIMIT $limit
            """

            result = list(
                memgraph_service.db.execute_and_fetch(query, {"limit": recent_limit})
            )

            recent_observations = []
            for row in result:
                recent_observations.append(
                    {
                        "created_at": row["created_at"],
                        "entity_name": row["entity_name"],
                        "content": row["content"],
                    }
                )

            logger.info(f"Retrieved {len(recent_observations)} recent observations")
            response["recent_observations"] = recent_observations

        except Exception as e:
            logger.error(f"Error retrieving recent observations: {e}")
            response["recent_observations"] = []

        logger.info(
            "Gentle refresh completed successfully",
            core_identity_loaded=response["core_identity"] is not None,
            shortterm_memories_count=len(response["shortterm_memories"]),
            recent_observations_count=len(response["recent_observations"]),
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Error in gentle_refresh: {e}", correlation_id=correlation_id)
        error_response = {
            "success": False,
            "error": f"Error in gentle_refresh: {e}",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)


def register_gentle_refresh_tools(mcp: FastMCP) -> None:
    """Register gentle_refresh tools with the MCP server."""
    logger = get_logger("tools.gentle_refresh")

    # Register the gentle_refresh tool
    mcp.tool(gentle_refresh)

    logger.debug("gentle_refresh tools registered")

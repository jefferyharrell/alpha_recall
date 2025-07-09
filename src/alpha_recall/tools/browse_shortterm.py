"""Browse shortterm tool for Alpha-Recall v1.0.0."""

import json
import time

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.redis import get_redis_service
from ..services.time import time_service
from ..utils.correlation import create_child_correlation_id, set_correlation_id

__all__ = ["browse_shortterm", "register_browse_shortterm_tool"]


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

    # Get Redis service
    redis_service = get_redis_service()
    client = redis_service.client

    try:
        # Calculate time range if 'since' is provided
        if since:
            try:
                # Try parsing as a duration string first (e.g., "6h", "2d", "1w")
                now = time_service.utc_now()

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
                    # Try parsing as absolute datetime with TimeService
                    parsed_time = time_service.parse(since)
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
        now = time_service.utc_now()

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
                    created_time = time_service.parse(created_at)
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


def register_browse_shortterm_tool(mcp: FastMCP) -> None:
    """Register the browse_shortterm tool with the MCP server."""
    logger = get_logger("tools.browse_shortterm")

    # Register the tool
    mcp.tool(browse_shortterm)

    logger.debug("browse_shortterm tool registered")

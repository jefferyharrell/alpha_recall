"""Browse narrative tool for chronological listing with filtering."""

import asyncio
import json
import time

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.factory import get_narrative_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["browse_narrative", "register_browse_narrative_tools"]


def browse_narrative(
    limit: int = 10,
    offset: int = 0,
    since: str | None = None,
    participants: list[str] | None = None,
    tags: list[str] | None = None,
    outcome: str | None = None,
) -> str:
    """List narrative stories chronologically with optional filtering.

    Args:
        limit: Maximum number of stories to return (default: 10)
        offset: Number of stories to skip for pagination (default: 0)
        since: Time duration to look back (e.g., "6h", "2d", "1w")
        participants: Filter by participants involved in stories
        tags: Filter by story tags
        outcome: Filter by story outcome status

    Returns:
        JSON string with paginated story list
    """
    correlation_id = generate_correlation_id("browse_narrative")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.browse_narrative")
    logger.info(
        "Browse narrative request received",
        limit=limit,
        offset=offset,
        since=since,
        participants=participants,
        tags=tags,
        outcome=outcome,
        correlation_id=correlation_id,
    )

    try:
        # Validate parameters
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        if offset < 0:
            raise ValueError("offset must be non-negative")

        # Clean up filter parameters
        clean_participants = [p.strip() for p in (participants or []) if p.strip()]
        clean_tags = [t.strip() for t in (tags or []) if t.strip()]
        clean_outcome = outcome.strip() if outcome else None

        # Perform the actual browse using NarrativeService
        start_time = time.time()
        narrative_service = get_narrative_service()

        # Call async service method from sync context
        browse_result = asyncio.run(
            narrative_service.list_stories(
                limit=limit,
                offset=offset,
                since=since,
                participants=clean_participants,
                tags=clean_tags,
                outcome=clean_outcome,
            )
        )
        query_time_ms = int((time.time() - start_time) * 1000)

        # Return response structure
        response = {
            "success": True,
            "browse_data": {
                "stories": browse_result["stories"],
                "pagination": {
                    "limit": browse_result["limit"],
                    "offset": browse_result["offset"],
                    "total_count": browse_result["total_count"],
                    "has_more": browse_result["has_more"],
                },
                "filters": {
                    "since": since,
                    "participants": clean_participants,
                    "tags": clean_tags,
                    "outcome": clean_outcome,
                },
                "metadata": {
                    "query_time_ms": query_time_ms,
                    "storage_location": "hybrid_redis_memgraph",
                    "sort_order": "chronological_desc",
                },
            },
            "correlation_id": correlation_id,
        }

        logger.info(
            "Narrative browse completed",
            results_count=browse_result["returned_count"],
            total_count=browse_result["total_count"],
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(
            "Failed to browse narratives",
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


def register_browse_narrative_tools(mcp: FastMCP) -> None:
    """Register browse_narrative tools with the MCP server."""
    logger = get_logger("tools.browse_narrative")

    # Register the browse_narrative tool
    mcp.tool(browse_narrative)

    logger.debug("Browse narrative tools registered")

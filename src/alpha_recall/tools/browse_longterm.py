"""Browse longterm tool for paginated entity listing."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["browse_longterm", "register_browse_longterm_tools"]


def browse_longterm(limit: int = 20, offset: int = 0) -> str:
    """Browse entities with pagination.

    Args:
        limit: Maximum number of entities to return (default 20)
        offset: Number of entities to skip for pagination (default 0)

    Returns:
        JSON string with paginated entity list
    """
    # Generate correlation ID for this operation
    correlation_id = generate_correlation_id("browse_ltm")
    set_correlation_id(correlation_id)

    tool_logger = get_logger("tools.browse_longterm")

    tool_logger.info(
        "Browse longterm requested",
        limit=limit,
        offset=offset,
        correlation_id=correlation_id,
    )

    try:
        memgraph_service = get_memgraph_service()

        # Test connection first
        if not memgraph_service.test_connection():
            raise Exception("Memgraph connection test failed")

        # Browse entities
        browse_data = memgraph_service.browse_entities(limit, offset)

        # Prepare response
        response_data = {
            "success": True,
            "browse_data": browse_data,
            "correlation_id": correlation_id,
        }

        response = json.dumps(response_data, indent=2, default=str)

        tool_logger.info(
            "Browse longterm completed successfully",
            limit=limit,
            offset=offset,
            results_count=browse_data.get("pagination", {}).get("results_count", 0),
            total_count=browse_data.get("pagination", {}).get("total_count", 0),
            has_more=browse_data.get("pagination", {}).get("has_more", False),
            correlation_id=correlation_id,
        )

        return response

    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": e.__class__.__name__,
            "correlation_id": correlation_id,
        }

        response = json.dumps(error_response, indent=2)

        tool_logger.error(
            "Browse longterm failed",
            limit=limit,
            offset=offset,
            error=str(e),
            error_type=e.__class__.__name__,
            correlation_id=correlation_id,
        )

        return response


def register_browse_longterm_tools(mcp: FastMCP) -> None:
    """Register browse_longterm tools with the MCP server."""
    logger = get_logger("tools.browse_longterm")

    # Register tools defined at module level
    mcp.tool(browse_longterm)

    logger.debug("browse_longterm tools registered")

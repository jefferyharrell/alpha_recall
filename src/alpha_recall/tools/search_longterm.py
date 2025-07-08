"""Long-term memory search tool."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["search_longterm", "register_search_longterm_tools"]


def search_longterm(query: str, entity: str | None = None, limit: int = 10) -> str:
    """Search long-term memory observations using semantic similarity.

    Args:
        query: The search query text
        entity: Optional entity name to filter results
        limit: Maximum number of results to return (default 10)

    Returns:
        JSON string with search results
    """
    # Generate correlation ID for this operation
    correlation_id = generate_correlation_id("ltm_search")
    set_correlation_id(correlation_id)

    tool_logger = get_logger("tools.search_longterm")

    tool_logger.info(
        "Long-term memory search requested",
        query=query,
        entity_filter=entity,
        limit=limit,
        correlation_id=correlation_id,
    )

    try:
        memgraph_service = get_memgraph_service()

        # Test connection first
        if not memgraph_service.test_connection():
            raise Exception("Memgraph connection test failed")

        # Search observations
        observations = memgraph_service.search_observations(query, entity, limit)

        # Prepare response
        response_data = {
            "success": True,
            "query": query,
            "entity_filter": entity,
            "limit": limit,
            "results_count": len(observations),
            "observations": observations,
            "correlation_id": correlation_id,
        }

        response = json.dumps(response_data, indent=2, default=str)

        tool_logger.info(
            "Long-term memory search completed successfully",
            query=query,
            entity_filter=entity,
            results_count=len(observations),
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
            "Long-term memory search failed",
            query=query,
            entity_filter=entity,
            error=str(e),
            error_type=e.__class__.__name__,
            correlation_id=correlation_id,
        )

        return response


def register_search_longterm_tools(mcp: FastMCP) -> None:
    """Register search_longterm tools with the MCP server."""
    logger = get_logger("tools.search_longterm")

    # Register tools defined at module level
    mcp.tool(search_longterm)

    logger.debug("search_longterm tools registered")

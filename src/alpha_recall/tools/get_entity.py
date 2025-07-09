"""Get entity tool for direct entity access with observations."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["get_entity", "register_get_entity_tools"]


def get_entity(entity_name: str) -> str:
    """Get entity information along with all its observations.

    Args:
        entity_name: The name of the entity to retrieve

    Returns:
        JSON string with entity data and all observations
    """
    # Generate correlation ID for this operation
    correlation_id = generate_correlation_id("get_entity")
    set_correlation_id(correlation_id)

    tool_logger = get_logger("tools.get_entity")

    tool_logger.info(
        "Get entity requested",
        entity_name=entity_name,
        correlation_id=correlation_id,
    )

    try:
        memgraph_service = get_memgraph_service()

        # Test connection first
        if not memgraph_service.test_connection():
            raise Exception("Memgraph connection test failed")

        # Get entity with observations
        entity_data = memgraph_service.get_entity_with_observations(entity_name)

        # Prepare response
        response_data = {
            "success": True,
            "entity": entity_data,
            "correlation_id": correlation_id,
        }

        response = json.dumps(response_data, indent=2, default=str)

        tool_logger.info(
            "Get entity completed successfully",
            entity_name=entity_name,
            observations_count=entity_data.get("observations_count", 0),
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
            "Get entity failed",
            entity_name=entity_name,
            error=str(e),
            error_type=e.__class__.__name__,
            correlation_id=correlation_id,
        )

        return response


def register_get_entity_tools(mcp: FastMCP) -> None:
    """Register get_entity tools with the MCP server."""
    logger = get_logger("tools.get_entity")

    # Register tools defined at module level
    mcp.tool(get_entity)

    logger.debug("get_entity tools registered")

"""Get relationships tool for entity relationship browsing."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["get_relationships", "register_get_relationships_tools"]


def get_relationships(entity_name: str) -> str:
    """Get all relationships for an entity (both incoming and outgoing).

    Args:
        entity_name: The name of the entity to get relationships for

    Returns:
        JSON string with all relationships for the entity
    """
    # Generate correlation ID for this operation
    correlation_id = generate_correlation_id("get_rel")
    set_correlation_id(correlation_id)

    tool_logger = get_logger("tools.get_relationships")

    tool_logger.info(
        "Get relationships requested",
        entity_name=entity_name,
        correlation_id=correlation_id,
    )

    try:
        memgraph_service = get_memgraph_service()

        # Test connection first
        if not memgraph_service.test_connection():
            raise Exception("Memgraph connection test failed")

        # Get entity relationships
        relationship_data = memgraph_service.get_entity_relationships(entity_name)

        # Prepare response
        response_data = {
            "success": True,
            "relationships": relationship_data,
            "correlation_id": correlation_id,
        }

        response = json.dumps(response_data, indent=2, default=str)

        tool_logger.info(
            "Get relationships completed successfully",
            entity_name=entity_name,
            total_relationships=relationship_data.get("total_relationships", 0),
            outgoing_count=len(relationship_data.get("outgoing_relationships", [])),
            incoming_count=len(relationship_data.get("incoming_relationships", [])),
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
            "Get relationships failed",
            entity_name=entity_name,
            error=str(e),
            error_type=e.__class__.__name__,
            correlation_id=correlation_id,
        )

        return response


def register_get_relationships_tools(mcp: FastMCP) -> None:
    """Register get_relationships tools with the MCP server."""
    logger = get_logger("tools.get_relationships")

    # Register tools defined at module level
    mcp.tool(get_relationships)

    logger.debug("get_relationships tools registered")

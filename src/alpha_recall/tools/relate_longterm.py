"""Long-term memory relationship creation tool."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["relate_longterm", "register_relate_longterm_tools"]


def relate_longterm(entity: str, to_entity: str, as_type: str) -> str:
    """Create a relationship between two entities in the knowledge graph.

    Args:
        entity: The source entity name
        to_entity: The target entity name
        as_type: The type of relationship (e.g., "works_with", "located_in", "instance_of")

    Returns:
        JSON string with operation result
    """
    # Generate correlation ID for this operation
    correlation_id = generate_correlation_id("ltm_rel")
    set_correlation_id(correlation_id)

    tool_logger = get_logger("tools.relate_longterm")

    tool_logger.info(
        "Long-term relationship creation requested",
        entity=entity,
        to_entity=to_entity,
        relationship_type=as_type,
        correlation_id=correlation_id,
    )

    try:
        memgraph_service = get_memgraph_service()

        # Test connection first
        if not memgraph_service.test_connection():
            raise Exception("Memgraph connection test failed")

        # Create relationship
        relationship_result = memgraph_service.create_relationship(
            entity, to_entity, as_type
        )

        # Prepare response
        response_data = {
            "success": True,
            "relationship": relationship_result,
            "correlation_id": correlation_id,
        }

        response = json.dumps(response_data, indent=2, default=str)

        tool_logger.info(
            "Long-term relationship creation completed successfully",
            entity=entity,
            to_entity=to_entity,
            relationship_type=as_type,
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
            "Long-term relationship creation failed",
            entity=entity,
            to_entity=to_entity,
            relationship_type=as_type,
            error=str(e),
            error_type=e.__class__.__name__,
            correlation_id=correlation_id,
        )

        return response


def register_relate_longterm_tools(mcp: FastMCP) -> None:
    """Register relate_longterm tools with the MCP server."""
    logger = get_logger("tools.relate_longterm")

    # Register tools defined at module level
    mcp.tool(relate_longterm)

    logger.debug("relate_longterm tools registered")

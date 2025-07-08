"""Long-term memory creation tool."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["remember_longterm", "register_remember_longterm_tools"]


def remember_longterm(
    entity: str, observation: str | None = None, type: str | None = None
) -> str:
    """Create or update an entity in the knowledge graph with optional observations.

    Args:
        entity: The name of the entity to create or update
        observation: Optional observation to add to the entity
        type: Optional type classification for the entity

    Returns:
        JSON string with operation result
    """
    # Generate correlation ID for this operation
    correlation_id = generate_correlation_id("ltm")
    set_correlation_id(correlation_id)

    tool_logger = get_logger("tools.remember_longterm")

    tool_logger.info(
        "Long-term memory creation requested",
        entity=entity,
        has_observation=observation is not None,
        observation_length=len(observation) if observation else 0,
        entity_type=type,
        correlation_id=correlation_id,
    )

    try:
        memgraph_service = get_memgraph_service()

        # Test connection first
        if not memgraph_service.test_connection():
            raise Exception("Memgraph connection test failed")

        # Create or update entity
        entity_result = memgraph_service.create_or_update_entity(entity, type)

        # Add observation if provided
        observation_result = None
        if observation:
            observation_result = memgraph_service.add_observation(entity, observation)

        # Prepare response
        response_data = {
            "success": True,
            "entity": entity_result,
            "observation": observation_result,
            "correlation_id": correlation_id,
        }

        response = json.dumps(response_data, indent=2, default=str)

        tool_logger.info(
            "Long-term memory creation completed successfully",
            entity=entity,
            entity_type=type,
            observation_added=observation is not None,
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
            "Long-term memory creation failed",
            entity=entity,
            entity_type=type,
            error=str(e),
            error_type=e.__class__.__name__,
            correlation_id=correlation_id,
        )

        return response


def register_remember_longterm_tools(mcp: FastMCP) -> None:
    """Register remember_longterm tools with the MCP server."""
    logger = get_logger("tools.remember_longterm")

    # Register tools defined at module level
    mcp.tool(remember_longterm)

    logger.debug("remember_longterm tools registered")

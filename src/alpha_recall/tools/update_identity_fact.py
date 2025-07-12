"""Update identity fact tool for dynamic identity management."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.redis import get_redis_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["update_identity_fact", "register_update_identity_fact_tools"]


def update_identity_fact(fact: str, new_score: float) -> str:
    """
    Update the position of an existing identity fact in Alpha's dynamic identity system.

    This tool allows Alpha to reorder her identity facts by changing their scores.
    The fact must already exist in the identity system.

    Args:
        fact: The exact identity fact to update (must match existing fact)
        new_score: New score for positioning (determines order in the list)

    Returns:
        JSON string with operation result including old/new scores and positions
    """
    correlation_id = generate_correlation_id("update_identity_fact")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.update_identity_fact")
    logger.info(
        "Updating identity fact",
        fact_length=len(fact),
        new_score=new_score,
        correlation_id=correlation_id,
    )

    try:
        # Validation
        if not fact or not fact.strip():
            return json.dumps(
                {
                    "success": False,
                    "error": "Identity fact cannot be empty",
                }
            )

        fact = fact.strip()

        # Score validation
        if new_score < 0:
            return json.dumps(
                {
                    "success": False,
                    "error": "Score must be non-negative",
                }
            )

        # Update the fact via Redis service
        redis_service = get_redis_service()
        result = redis_service.update_identity_fact(fact, new_score)

        logger.info(
            "Identity fact update completed",
            success=result.get("success", False),
            old_score=result.get("old_score"),
            new_score=result.get("new_score"),
            old_position=result.get("old_position"),
            new_position=result.get("new_position"),
            correlation_id=correlation_id,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(
            f"Error in update_identity_fact: {e}", correlation_id=correlation_id
        )
        error_response = {
            "success": False,
            "error": f"Error updating identity fact: {e}",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)


def register_update_identity_fact_tools(mcp: FastMCP) -> None:
    """Register update_identity_fact tools with the MCP server."""
    logger = get_logger("tools.update_identity_fact")

    # Register the update_identity_fact tool
    mcp.tool(update_identity_fact)

    logger.debug("update_identity_fact tools registered")

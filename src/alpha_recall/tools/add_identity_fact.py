"""Add identity fact tool for dynamic identity management."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.redis import get_redis_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["add_identity_fact", "register_add_identity_fact_tools"]


def add_identity_fact(fact: str, score: float = None) -> str:
    """
    Add or update an identity fact in Alpha's dynamic identity system.

    This tool allows Alpha to manage her own identity documentation through
    an ordered set of facts stored in Redis. Facts are automatically appended
    to the end (auto-scoring) unless an explicit score is provided for positioning.

    Args:
        fact: The identity fact to add (e.g., "Alpha adopted female gender identity on July 12, 2025")
        score: Optional score for positioning (defaults to auto-append after highest score)

    Returns:
        JSON string with operation result including success status, fact, score, position, and metadata
    """
    correlation_id = generate_correlation_id("add_identity_fact")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.add_identity_fact")
    logger.info(
        "Adding identity fact",
        fact_length=len(fact),
        score=score,
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

        # Length validation (reasonable limit)
        if len(fact) > 1000:
            return json.dumps(
                {
                    "success": False,
                    "error": "Identity fact is too long (maximum 1000 characters)",
                }
            )

        # Score validation
        if score is not None and score < 0:
            return json.dumps(
                {
                    "success": False,
                    "error": "Score must be non-negative",
                }
            )

        # Add the fact via Redis service
        redis_service = get_redis_service()
        result = redis_service.add_identity_fact(fact, score)

        logger.info(
            "Identity fact operation completed",
            success=result.get("success", False),
            is_update=result.get("updated", False),
            score=result.get("score"),
            position=result.get("position"),
            correlation_id=correlation_id,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error in add_identity_fact: {e}", correlation_id=correlation_id)
        error_response = {
            "success": False,
            "error": f"Error adding identity fact: {e}",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)


def register_add_identity_fact_tools(mcp: FastMCP) -> None:
    """Register add_identity_fact tools with the MCP server."""
    logger = get_logger("tools.add_identity_fact")

    # Register the add_identity_fact tool
    mcp.tool(add_identity_fact)

    logger.debug("add_identity_fact tools registered")

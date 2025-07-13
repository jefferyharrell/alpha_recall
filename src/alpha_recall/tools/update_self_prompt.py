"""
Self-prompt management tool.

This tool manages Alpha's dynamic self-prompt - a self-written prompt addendum
that gets injected into gentle_refresh output. Can be used for project continuity,
temporary mindset shifts, philosophical framings, or any epistolary communication
with future instances of self.
"""

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.redis import get_redis_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["update_self_prompt", "register_update_self_prompt_tools"]


async def update_self_prompt(message: str) -> str:
    """
    Update Alpha's dynamic self-prompt for context injection.

    This stores a self-written prompt addendum that gets displayed in gentle_refresh
    output. Can be used for project continuity, cognitive frameworks, mindset shifts,
    philosophical orientations, or any epistolary communication with future self.
    Approach it like writing a letter to yourself.

    Args:
        message: Natural language self-prompt content. Use empty string to clear.

    Returns:
        JSON string with operation result
    """
    import json

    logger = get_logger("tools.update_self_prompt")
    correlation_id = generate_correlation_id("update_self_prompt")
    set_correlation_id(correlation_id)

    logger.info("Update self prompt called", message_length=len(message))

    try:
        redis_service = get_redis_service()
        result = redis_service.update_self_prompt(message)

        if result["success"]:
            logger.info(
                "Intentional memory updated successfully",
                operation=result.get("operation", "unknown"),
                correlation_id=correlation_id,
            )
        else:
            logger.error(
                "Failed to update intentional memory",
                error=result.get("error"),
                correlation_id=correlation_id,
            )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error("Update self prompt failed", error=str(e))
        error_result = {"success": False, "error": f"Tool execution failed: {e}"}
        return json.dumps(error_result, indent=2)


def register_update_self_prompt_tools(mcp: FastMCP) -> None:
    """Register self-prompt tools with the MCP server."""
    logger = get_logger("tools.update_self_prompt")

    mcp.tool(update_self_prompt)

    logger.debug("Self-prompt tools registered")

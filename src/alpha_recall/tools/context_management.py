"""
Modular context management tools for Alpha-Recall MCP server.

These tools provide fine-grained control over context injection blocks that appear
in gentle_refresh output, allowing dynamic management of self-prompts, autobiographical
content, project notes, and other contextual information.
"""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.redis import get_redis_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = [
    "set_context_block",
    "get_context_block",
    "list_context_blocks",
    "delete_context_block",
    "register_context_management_tools",
]


async def set_context_block(key: str, content: str) -> str:
    """
    Set a context block for modular self-prompt management.

    Context blocks are arbitrary Markdown sections that get injected into gentle_refresh
    output. Use empty content to remove a block.

    Args:
        key: The context block key (e.g., 'autobiography', 'current_project', 'system_prompt_experiment')
        content: The Markdown content to store. Use empty string to remove the block.

    Returns:
        JSON string with operation result
    """
    logger = get_logger("tools.set_context_block")
    correlation_id = generate_correlation_id("set_context_block")
    set_correlation_id(correlation_id)

    try:
        logger.info(
            "Setting context block",
            key=key,
            content_length=len(content),
            correlation_id=correlation_id,
        )

        redis_service = get_redis_service()
        result = redis_service.set_context_block(key, content)

        logger.info(
            "Context block operation completed",
            success=result.get("success"),
            operation=result.get("operation", "unknown"),
            correlation_id=correlation_id,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(
            "Error in set_context_block tool",
            error=str(e),
            key=key,
            correlation_id=correlation_id,
        )
        return json.dumps(
            {"success": False, "error": f"Tool execution failed: {e}"}, indent=2
        )


async def get_context_block(key: str) -> str:
    """
    Get a specific context block.

    Args:
        key: The context block key to retrieve

    Returns:
        JSON string with context block content
    """
    logger = get_logger("tools.get_context_block")
    correlation_id = generate_correlation_id("get_context_block")
    set_correlation_id(correlation_id)

    try:
        logger.info("Getting context block", key=key, correlation_id=correlation_id)

        redis_service = get_redis_service()
        result = redis_service.get_context_block(key)

        logger.info(
            "Context block retrieved",
            success=result.get("success"),
            has_content=result.get("has_content"),
            correlation_id=correlation_id,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(
            "Error in get_context_block tool",
            error=str(e),
            key=key,
            correlation_id=correlation_id,
        )
        return json.dumps(
            {"success": False, "error": f"Tool execution failed: {e}"}, indent=2
        )


async def list_context_blocks() -> str:
    """
    List all available context blocks.

    Returns:
        JSON string with list of context block keys
    """
    logger = get_logger("tools.list_context_blocks")
    correlation_id = generate_correlation_id("list_context_blocks")
    set_correlation_id(correlation_id)

    try:
        logger.info("Listing context blocks", correlation_id=correlation_id)

        redis_service = get_redis_service()
        result = redis_service.list_context_blocks()

        logger.info(
            "Context blocks listed",
            success=result.get("success"),
            count=result.get("count", 0),
            correlation_id=correlation_id,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(
            "Error in list_context_blocks tool",
            error=str(e),
            correlation_id=correlation_id,
        )
        return json.dumps(
            {"success": False, "error": f"Tool execution failed: {e}"}, indent=2
        )


async def delete_context_block(key: str) -> str:
    """
    Delete a specific context block.

    Args:
        key: The context block key to delete

    Returns:
        JSON string with deletion result
    """
    logger = get_logger("tools.delete_context_block")
    correlation_id = generate_correlation_id("delete_context_block")
    set_correlation_id(correlation_id)

    try:
        logger.info("Deleting context block", key=key, correlation_id=correlation_id)

        redis_service = get_redis_service()
        result = redis_service.delete_context_block(key)

        logger.info(
            "Context block deletion completed",
            success=result.get("success"),
            key_existed=result.get("key_existed"),
            correlation_id=correlation_id,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(
            "Error in delete_context_block tool",
            error=str(e),
            key=key,
            correlation_id=correlation_id,
        )
        return json.dumps(
            {"success": False, "error": f"Tool execution failed: {e}"}, indent=2
        )


def register_context_management_tools(mcp: FastMCP) -> None:
    """Register all context management tools with the MCP server."""
    logger = get_logger("tools.context_management")

    mcp.tool(set_context_block)
    mcp.tool(get_context_block)
    mcp.tool(list_context_blocks)
    mcp.tool(delete_context_block)

    logger.info("Context management tools registered successfully")

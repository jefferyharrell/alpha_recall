"""Alpha-Recall MCP Server implementation."""

import os

from fastmcp import FastMCP

from .config import settings
from .logging import configure_logging, get_logger
from .tools import (
    register_browse_longterm_tools,
    register_browse_narrative_tools,
    register_browse_shortterm_tool,
    register_consolidate_shortterm_tools,
    register_gentle_refresh_tools,
    register_get_entity_tools,
    register_get_relationships_tools,
    register_health_tools,
    register_recall_narrative_tools,
    register_relate_longterm_tools,
    register_remember_longterm_tools,
    register_remember_narrative_tools,
    register_remember_shortterm_tool,
    register_search_all_memories_tools,
    register_search_longterm_tools,
    register_search_narratives_tools,
    register_search_shortterm_tool,
)
from .version import __version__


def create_server():
    """Create and configure the FastMCP server."""
    logger = get_logger("server")

    mcp = FastMCP("alpha-recall")
    logger.debug("FastMCP instance created")

    # Register all tool modules
    register_health_tools(mcp)
    register_gentle_refresh_tools(mcp)
    register_remember_shortterm_tool(mcp)
    register_browse_shortterm_tool(mcp)
    register_search_shortterm_tool(mcp)
    register_consolidate_shortterm_tools(mcp)
    register_remember_longterm_tools(mcp)
    register_relate_longterm_tools(mcp)
    register_search_longterm_tools(mcp)
    register_search_all_memories_tools(mcp)
    register_get_entity_tools(mcp)
    register_get_relationships_tools(mcp)
    register_browse_longterm_tools(mcp)
    register_remember_narrative_tools(mcp)
    register_search_narratives_tools(mcp)
    register_recall_narrative_tools(mcp)
    register_browse_narrative_tools(mcp)

    logger.debug("All tools registered")
    return mcp


def main():
    """Main entry point for the Alpha-Recall MCP server."""
    # Initialize logging first
    logger = configure_logging()

    transport = settings.mcp_transport
    host = settings.host
    port = settings.port

    # Set environment variables for FastMCP to pick up
    if host != "localhost":  # Only override if different from default
        os.environ["FASTMCP_HOST"] = host
    if port is not None:
        os.environ["FASTMCP_PORT"] = str(port)

    logger.info(
        "Starting Alpha-Recall",
        version=__version__,
        transport=transport,
        host=host,
        port=port or "default",
        log_format=settings.log_format,
        log_level=settings.log_level,
    )

    try:
        # Create and run server
        mcp = create_server()
        logger.info("Server created successfully", server_name="alpha-recall")

        mcp.run(transport=transport, host=host, port=port)
    except Exception as e:
        logger.error(
            "Failed to start server",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise


if __name__ == "__main__":
    main()

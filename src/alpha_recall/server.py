"""Alpha-Recall MCP Server implementation."""

import os
from fastmcp import FastMCP
from .config import settings
from .logging import configure_logging, get_logger
from .tools import register_health_tools
from .version import __version__

def create_server():
    """Create and configure the FastMCP server."""
    logger = get_logger("server")
    
    mcp = FastMCP("alpha-recall")
    logger.debug("FastMCP instance created")
    
    # Register all tool modules
    register_health_tools(mcp)
    
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
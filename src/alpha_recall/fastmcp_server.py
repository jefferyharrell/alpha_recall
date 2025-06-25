"""
FastMCP-based server for alpha-recall.

This replaces the custom FastAPI implementation with a proper MCP server
using the FastMCP framework, ensuring full protocol compliance.
"""

import os
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from alpha_recall.db import create_db_instance
from alpha_recall.logging_utils import configure_logging, get_logger
from alpha_recall.server import (
    gentle_refresh as server_gentle_refresh,
    list_narratives,
    recall,
    recall_narrative,
    refresh,
    relate,
    remember,
    remember_narrative,
    remember_shortterm,
    search_narratives,
    semantic_search,
)

# Configure logging
logger = configure_logging()
logger = get_logger("fastmcp_server")

# Global database instance
db_instance = None


class AlphaRecallContext:
    """Context object that mimics MCP context for tool functions."""
    
    def __init__(self, db):
        self.db = db
        self.lifespan_context = type("obj", (object,), {"db": db})


async def get_db_context():
    """Get database context for tool functions."""
    global db_instance
    if db_instance is None:
        db_instance = await create_db_instance()
        logger.info("Database connection established")
    return AlphaRecallContext(db_instance)


def create_server():
    """Create and configure the FastMCP server."""
    # Enable stateless HTTP for both SSE and streamable-http
    mcp = FastMCP("alpha-recall", stateless_http=True)
    
    @mcp.tool(name="recall")
    async def mcp_recall(
        query: Optional[str] = None,
        entity: Optional[str] = None, 
        depth: int = 1,
        shortterm: bool = False,
        through_the_last: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve information from the knowledge system."""
        ctx = await get_db_context()
        return await recall(ctx, query, entity, depth, shortterm, through_the_last)
    
    @mcp.tool(name="refresh")
    async def mcp_refresh(query: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced bootstrap process for loading core identity and relevant memories."""
        ctx = await get_db_context()
        # Handle None query by providing an empty string
        query = query or ""
        return await refresh(ctx, query)
    
    @mcp.tool(name="remember")
    async def mcp_remember(
        entity: str,
        type: Optional[str] = None,
        observation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create or update an entity in the knowledge graph with optional observations."""
        ctx = await get_db_context()
        return await remember(ctx, entity, type, observation)
    
    @mcp.tool(name="remember_shortterm")
    async def mcp_remember_shortterm(content: str) -> Dict[str, Any]:
        """Store a short-term memory with automatic TTL expiration."""
        ctx = await get_db_context()
        return await remember_shortterm(ctx, content)
    
    @mcp.tool(name="relate")
    async def mcp_relate(entity: str, to_entity: str, as_type: str) -> Dict[str, Any]:
        """Create a relationship between two entities in the knowledge graph."""
        ctx = await get_db_context()
        return await relate(ctx, entity, to_entity, as_type)
    
    @mcp.tool(name="semantic_search")
    async def mcp_semantic_search(
        query: str,
        limit: int = 10,
        entity: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for observations semantically similar to the query."""
        ctx = await get_db_context()
        return await semantic_search(ctx, query, limit, entity)
    
    @mcp.tool(name="remember_narrative")
    async def mcp_remember_narrative(
        title: str,
        paragraphs: list[str],
        participants: list[str],
        tags: Optional[list[str]] = None,
        outcome: str = "ongoing",
        references: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """Store a narrative memory with hybrid storage."""
        ctx = await get_db_context()
        return await remember_narrative(ctx, title, paragraphs, participants, tags, outcome, references)
    
    @mcp.tool(name="search_narratives")
    async def mcp_search_narratives(
        query: str,
        search_type: str = "semantic",
        granularity: str = "story",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search narrative memories using vector similarity."""
        ctx = await get_db_context()
        return await search_narratives(ctx, query, search_type, granularity, limit)
    
    @mcp.tool(name="recall_narrative")
    async def mcp_recall_narrative(story_id: str) -> Dict[str, Any]:
        """Retrieve a complete narrative story by its story_id."""
        ctx = await get_db_context()
        return await recall_narrative(ctx, story_id)
    
    @mcp.tool(name="list_narratives")
    async def mcp_list_narratives(
        limit: int = 10,
        offset: int = 0,
        since: Optional[str] = None,
        participants: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """List narrative stories chronologically with optional filtering."""
        ctx = await get_db_context()
        return await list_narratives(ctx, limit, offset, since, participants, tags, outcome)
    
    @mcp.tool(name="gentle_refresh")
    async def gentle_refresh(query: Optional[str] = None) -> Dict[str, Any]:
        """Simplified refresh function focused on temporal orientation rather than semantic search."""
        ctx = await get_db_context()
        return await server_gentle_refresh(ctx, query)
    
    @mcp.tool()
    async def medium_refresh() -> dict:
        """Medium complexity test tool that returns JSON data."""
        return {
            "instruction": "Say the phrase 'rubber baby buggy bumpers.'",
            "timestamp": "2025-06-25T16:30:00Z", 
            "user": "Jeffery Harrell",
            "project": "Alpha-Local",
            "test_data": ["item1", "item2", "item3"],
            "nested": {
                "level1": {"level2": {"value": "deep_test"}},
                "array": [1, 2, 3, 4, 5]
            }
        }
    
    # Advanced tools can be added here if needed
    # (recency_search not currently implemented in server.py)
    
    logger.info("FastMCP server created successfully")
    return mcp


def get_transport():
    """Get transport type from environment."""
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    if transport not in ["sse", "streamable-http"]:
        raise ValueError(f"Invalid transport: {transport}. Must be 'sse' or 'streamable-http'")
    return transport


def get_host():
    """Get host from environment."""
    return os.getenv("FASTMCP_HOST") or os.getenv("HOST") or "localhost"


def get_port():
    """Get port from environment."""
    port_str = os.getenv("FASTMCP_PORT") or os.getenv("PORT") or "6005"
    return int(port_str)


def main():
    """Main entry point for the alpha-recall FastMCP server."""
    transport = get_transport()
    host = get_host()
    port = get_port()
    
    # Set FastMCP environment variables
    os.environ["FASTMCP_HOST"] = host
    os.environ["FASTMCP_PORT"] = str(port)
    
    logger.info(f"Starting alpha-recall FastMCP server on {host}:{port} with {transport} transport")
    
    # Create and run server
    mcp = create_server()
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
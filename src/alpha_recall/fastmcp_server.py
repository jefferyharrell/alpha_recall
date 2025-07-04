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
from alpha_recall.reminiscer import ReminiscerAgent
from alpha_recall.server import (
    ask_memory,
    gentle_refresh as server_gentle_refresh,
    list_narratives,
    recall_narrative,
    refresh,
    relate_longterm,
    remember_longterm,
    remember_narrative,
    remember_shortterm,
    search_all_memories,
    search_longterm,
    search_narratives,
    search_shortterm,
)

# Configure logging
logger = configure_logging()
logger = get_logger("fastmcp_server")

# Global instances
db_instance = None
reminiscer_instance = None


class AlphaRecallContext:
    """Context object that mimics MCP context for tool functions."""
    
    def __init__(self, db, reminiscer=None):
        self.db = db
        self.reminiscer = reminiscer
        # Create a proper context object with get method and subscriptable access
        class LifespanContext:
            def __init__(self, db, reminiscer):
                self._data = {"db": db, "reminiscer": reminiscer}
            def get(self, key, default=None):
                return self._data.get(key, default)
            def __getitem__(self, key):
                return self._data[key]
        self.lifespan_context = LifespanContext(db, reminiscer)


async def get_db_context():
    """Get database context for tool functions."""
    global db_instance, reminiscer_instance
    if db_instance is None:
        db_instance = await create_db_instance()
        logger.info("Database connection established")
        
        # Initialize reminiscer if enabled
        reminiscer_enabled = os.environ.get("REMINISCER_ENABLED", "false").lower() == "true"
        logger.info(f"Reminiscer enabled check: REMINISCER_ENABLED='{os.environ.get('REMINISCER_ENABLED', 'NOT_SET')}', reminiscer_enabled={reminiscer_enabled}")
        if reminiscer_enabled:
            try:
                model_name = os.environ.get("REMINISCER_MODEL", "llama3.1:8b")
                ollama_host = os.environ.get("REMINISCER_OLLAMA_HOST", "localhost")
                ollama_port = int(os.environ.get("REMINISCER_OLLAMA_PORT", "11434"))
                
                reminiscer_instance = ReminiscerAgent(
                    composite_db=db_instance,
                    model_name=model_name,
                    ollama_host=ollama_host,
                    ollama_port=ollama_port
                )
                logger.info(f"Reminiscer initialized with model {model_name} at {ollama_host}:{ollama_port}")
            except Exception as e:
                logger.warning(f"Failed to initialize reminiscer: {e}")
                reminiscer_instance = None
        
    return AlphaRecallContext(db_instance, reminiscer_instance)


def create_server():
    """Create and configure the FastMCP server."""
    # Enable stateless HTTP for both SSE and streamable-http
    mcp = FastMCP("alpha-recall", stateless_http=True)
    
    # @mcp.tool(name="refresh")
    # async def mcp_refresh(query: Optional[str] = None) -> Dict[str, Any]:
    #     """Enhanced bootstrap process for loading core identity and relevant memories."""
    #     ctx = await get_db_context()
    #     # Handle None query by providing an empty string
    #     query = query or ""
    #     return await refresh(ctx, query)
    
    @mcp.tool(name="remember_longterm")
    async def mcp_remember_longterm(
        entity: str,
        type: Optional[str] = None,
        observation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create or update an entity in the knowledge graph with optional observations."""
        ctx = await get_db_context()
        return await remember_longterm(ctx, entity, type, observation)
    
    @mcp.tool(name="remember_shortterm")
    async def mcp_remember_shortterm(content: str) -> Dict[str, Any]:
        """Store a short-term memory with automatic TTL expiration."""
        ctx = await get_db_context()
        return await remember_shortterm(ctx, content)
    
    @mcp.tool(name="relate_longterm")
    async def mcp_relate_longterm(entity: str, to_entity: str, as_type: str) -> Dict[str, Any]:
        """Create a relationship between two entities in the knowledge graph."""
        ctx = await get_db_context()
        return await relate_longterm(ctx, entity, to_entity, as_type)
    
    @mcp.tool(name="search_shortterm")
    async def mcp_search_shortterm(
        query: str,
        limit: int = 10,
        search_type: str = "semantic",
        through_the_last: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search short-term memories using semantic or emotional similarity."""
        ctx = await get_db_context()
        return await search_shortterm(ctx, query, limit, search_type, through_the_last)
    
    @mcp.tool(name="search_longterm")
    async def mcp_search_longterm(
        query: str,
        limit: int = 10,
        entity: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search long-term memory observations using semantic similarity."""
        ctx = await get_db_context()
        return await search_longterm(ctx, query, limit, entity)
    
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
    
    @mcp.tool(name="search_all_memories")
    async def search_all_memories_tool(
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search across all memory systems (STM, LTM, NM) with unified results."""
        ctx = await get_db_context()
        return await search_all_memories(ctx, query, limit, offset)
    
    @mcp.tool(name="ask_memory")
    async def ask_memory_tool(question: str, new_chat: bool = False) -> Dict[str, Any]:
        """Ask a conversational question to Alpha-Reminiscer about memories."""
        ctx = await get_db_context()
        return await ask_memory(ctx, question, new_chat)
    
    
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
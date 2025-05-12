"""
MCP server implementation for alpha_recall.

This module implements the Model Context Protocol (MCP) server for alpha_recall,
providing a structured memory system through Neo4j.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Dict, List, Optional, Any, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context

from alpha_recall.db import GraphDatabase, create_db_instance
from alpha_recall.logging_utils import configure_logging, get_logger
from alpha_recall.models.entities import Entity, Observation, Relationship

# Load environment variables
load_dotenv()

# Configure logging
logger = configure_logging()
logger = get_logger("server")

# Server name
SERVER_NAME = "alpha-recall"


@asynccontextmanager
async def server_lifespan(mcp_server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """
    Manage server startup and shutdown lifecycle.
    
    This context manager handles:
    - Initializing the database connection on startup
    - Closing the database connection on shutdown
    
    Args:
        mcp_server: The MCP server instance
        
    Yields:
        A context dictionary containing resources for tool handlers
    """
    logger.info(f"Starting {SERVER_NAME} MCP server")
    
    try:
        # Initialize database connection
        db = await create_db_instance()
        logger.info("Database connection established")
        
        # Yield the lifespan context with the database connection
        # Also set db directly on the mcp_server for tools that access ctx.db
        context = {"db": db}
        mcp_server.db = db
        yield context
        
    except Exception as e:
        logger.error(f"Error during server startup: {str(e)}")
        raise
    
    finally:
        # Clean up resources on shutdown
        db = mcp_server.lifespan_context.get("db")
        if db:
            logger.info("Closing database connection")
            await db.close()
        
        logger.info(f"Shutting down {SERVER_NAME} MCP server")


# Create MCP server with lifespan management
mcp = FastMCP(SERVER_NAME, lifespan=server_lifespan)


@mcp.tool(name="testing_recall")
async def recall(ctx: Context, entity: Optional[str] = None, depth: int = 1) -> Dict[str, Any]:
    """
    Retrieve information about an entity from the knowledge graph.
    
    If no entity is provided, returns a list of important entities.
    
    Args:
        ctx: The request context containing lifespan resources
        entity: Optional name of the entity to retrieve
        depth: How many relationship hops to include
            0: Only the entity itself
            1: Entity and direct relationships
            2+: Entity and extended network
            
    Returns:
        Dictionary containing the entity information and relationships
    """
    logger.info(f"Recall tool called: entity='{entity}', depth={depth}")
    
    # For debugging purposes, log the context object structure
    logger.debug(f"Context object: {type(ctx).__name__}")
    logger.debug(f"Context attributes: {dir(ctx)}")
    
    # Create a mock response for testing
    # This allows us to verify the tool is working even without database access
    mock_response = {
        "entity": entity or "NEXUS",
        "type": "Entity",
        "observations": ["This is a mock observation for testing purposes."],
        "relationships": [],
        "success": True,
        "note": "This is a mock response for testing. Database connection not available."
    }
    
    # Return the mock response for now
    return mock_response


@mcp.tool(name="testing_remember")
async def remember(ctx: Context, entity: str, entity_type: Optional[str] = None, observation: Optional[str] = None) -> Dict[str, Any]:
    """
    Create or update an entity in the knowledge graph with optional observations.
    
    Args:
        ctx: The request context containing lifespan resources
        entity: The name of the entity to create or update
        type: Optional type of entity (Person, Place, Concept, etc.)
        observation: Optional fact or observation about the entity
        
    Returns:
        Dictionary containing the created/updated entity information
    """
    logger.info(f"Remember tool called: entity='{entity}', type='{entity_type}', observation='{observation}'")
    
    # For debugging purposes, log the context object structure
    logger.debug(f"Context object: {type(ctx).__name__}")
    logger.debug(f"Context attributes: {dir(ctx)}")
    
    # Create a mock response for testing
    # This allows us to verify the tool is working even without database access
    mock_response = {
        "entity": entity,
        "type": entity_type or "Entity",
        "observation": observation,
        "success": True,
        "note": "This is a mock response for testing. Database connection not available."
    }
    
    # Return the mock response for now
    return mock_response


@mcp.tool(name="testing_relate")
async def relate(ctx: Context, entity: str, to_entity: str, as_type: str) -> Dict[str, Any]:
    """
    Create a relationship between two entities in the knowledge graph.
    
    Args:
        ctx: The request context containing lifespan resources
        entity: The source entity name
        to_entity: The target entity name
        as_type: The type of relationship (has, knows, located_in, etc.)
        
    Returns:
        Dictionary containing the created relationship information
    """
    logger.info(f"Relate tool called: entity='{entity}', to_entity='{to_entity}', as_type='{as_type}'")
    
    # For debugging purposes, log the context object structure
    logger.debug(f"Context object: {type(ctx).__name__}")
    logger.debug(f"Context attributes: {dir(ctx)}")
    
    # Create a mock response for testing
    # This allows us to verify the tool is working even without database access
    mock_response = {
        "entity": entity,
        "to_entity": to_entity,
        "as_type": as_type,
        "success": True,
        "note": "This is a mock response for testing. Database connection not available."
    }
    
    # Return the mock response for now
    return mock_response


async def main():
    """
    Main entry point for the MCP server.
    """
    try:
        # Run the MCP server with stdio
        # This is the standard way to run an MCP server
        await mcp.run_stdio()
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the server
    asyncio.run(main())

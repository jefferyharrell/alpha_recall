"""
MCP server implementation for alpha_recall.

This module implements the Model Context Protocol (MCP) server for alpha_recall,
providing a structured memory system through Neo4j with semantic search capabilities
via Qdrant and sentence-transformers.
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

# Determine mode (advanced tools exposed only if MODE=advanced)
MODE = os.environ.get("MODE", "").lower()

# Advanced tools (only register if MODE=advanced)
if MODE == "advanced":
    @mcp.tool(name="delete_entity")
    async def delete_entity(ctx: Context, entity: str) -> Dict[str, Any]:
        """
        Delete an entity and all its relationships (and attached observations) from the knowledge graph.
        Args:
            ctx: The request context containing lifespan resources
            entity: The name of the entity to delete
        Returns:
            Dictionary containing the deletion status and details
        """
        logger.info(f"[ADVANCED] Delete entity tool called: entity='{entity}'")
        db = None
        if hasattr(ctx, 'lifespan_context') and hasattr(ctx.lifespan_context, 'db'):
            db = ctx.lifespan_context.db
        elif hasattr(ctx, 'db'):
            db = ctx.db
        elif hasattr(mcp, 'db'):
            db = mcp.db
        if db is None:
            logger.error("Database connection not available for delete_entity")
            return {"error": "Database connection not available", "success": False}
        try:
            result = await db.delete_entity(entity)
            logger.info(f"[ADVANCED] Entity deletion result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error deleting entity: {str(e)}")
            return {"error": f"Error deleting entity: {str(e)}", "success": False}
            
@mcp.tool(name="semantic_search")
async def semantic_search(ctx: Context, query: str, limit: int = 10, entity: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for observations semantically similar to the query.
    
    Args:
        ctx: The request context containing lifespan resources
        query: The natural language query to search for
        limit: Maximum number of results to return
        entity: Optional entity name to filter results
        
    Returns:
        Dictionary containing the search results
    """
    logger.info(f"Semantic search tool called: query='{query}', entity='{entity}', limit={limit}")
    
    # Try to get the database connection from various places
    db = None
    if hasattr(ctx, 'lifespan_context') and hasattr(ctx.lifespan_context, 'db'):
        db = ctx.lifespan_context.db
    elif hasattr(ctx, 'db'):
        db = ctx.db
    elif hasattr(mcp, 'db'):
        db = mcp.db
        
    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available for semantic_search")
        return {
            "error": "Database connection not available",
            "success": False
        }
        
    try:
        # Check if db has semantic_search method (CompositeDatabase)
        if hasattr(db, 'semantic_search'):
            results = await db.semantic_search(query=query, limit=limit, entity_name=entity)
            return {
                "query": query,
                "results": results,
                "success": True
            }
        else:
            logger.error("Database does not support semantic search")
            return {
                "error": "Database does not support semantic search",
                "success": False
            }
    except Exception as e:
        logger.error(f"Error performing semantic search: {str(e)}")
        return {
            "error": f"Error performing semantic search: {str(e)}",
            "success": False
        }


@mcp.tool(name="recall")
async def recall(ctx: Context, query: Optional[str] = None, entity: Optional[str] = None, depth: int = 1) -> Dict[str, Any]:
    """
    Retrieve information about an entity from the knowledge graph.
    
    This enhanced version accepts a query parameter that can be either:
    1. An entity name - Returns the entity with its relationships
    2. A semantic search query - Returns semantically similar observations
    3. Both - Returns the entity and semantically similar observations
    
    If no query is provided, returns a list of important entities.
    
    Args:
        ctx: The request context containing lifespan resources
        query: Optional query string (entity name or semantic search query)
        depth: How many relationship hops to include for entity matches
            0: Only the entity itself
            1: Entity and direct relationships
            2+: Entity and extended network
            
    Returns:
        Dictionary containing:
        - exact_match: Entity information if query matches an entity name
        - semantic_results: Top semantically similar observations
    """
    logger.info(f"Enhanced recall tool called: query='{query}', entity='{entity}', depth={depth}")
    
    # If entity is provided but query is not, use entity as the query
    if entity and not query:
        query = entity
    
    # Try to get the database connection from various places
    db = None
    
    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, 'lifespan_context') and hasattr(ctx.lifespan_context, 'db'):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, 'db'):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, 'db'):
        db = mcp.db
    
    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available")
        return {
            "error": "Database connection not available",
            "success": False
        }
    
    try:
        exact_match = None
        semantic_results = []
        
        # Empty query returns important entities (bootstrap mode)
        # This behavior is maintained for backward compatibility
        if not query or query.strip() == "":
            bootstrap_node = os.environ.get("BOOTSTRAP_NODE", "Alpha")
            logger.info(f"Bootstrap mode activated, returning {bootstrap_node}")
            exact_match = await db.get_entity(bootstrap_node, depth)
        else:
            # First, try to get an exact entity match
            exact_match = await db.get_entity(query, depth)
            
            # Next, perform semantic search regardless of whether we found an exact match
            # Determine the number of semantic results to include
            semantic_limit = 5  # Default number of semantic results
            
            # Check if db has semantic_search method (CompositeDatabase)
            if hasattr(db, 'semantic_search'):
                semantic_results = await db.semantic_search(query=query, limit=semantic_limit)
            else:
                logger.warning("Database does not support semantic search")
        
        # Construct the combined response
        response = {
            "query": query,
            "success": True
        }
        
        # Add exact match if found
        if exact_match:
            response["exact_match"] = exact_match
        
        # Add semantic results if available
        if semantic_results:
            response["semantic_results"] = semantic_results
        
        # If neither exact match nor semantic results, return an error
        if not exact_match and not semantic_results:
            logger.warning(f"No results found for query: {query}")
            return {
                "query": query,
                "error": f"No results found for query: '{query}'",
                "success": False
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced recall: {str(e)}")
        return {
            "query": query,
            "error": f"Error in enhanced recall: {str(e)}",
            "success": False
        }


@mcp.tool(name="refresh")
async def refresh(ctx: Context, query: str) -> Dict[str, Any]:
    """
    Implements the two-tiered bootstrap process as described in ADR-008.
    
    This function loads:
    1. Tier 1: Core identity information (essential bootstrap entity)
    2. Tier 2: Contextually relevant memories based on the user's greeting
    
    Args:
        ctx: The request context containing lifespan resources
        query: The user's greeting or first message (used for semantic search)
        
    Returns:
        Dictionary containing:
        - core_identity: The core identity entity (Tier 1)
        - relevant_memories: Semantically relevant observations based on the query (Tier 2)
    """
    logger.info(f"Refresh tool called with query: '{query}'")
    
    # Try to get the database connection from various places
    db = None
    
    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, 'lifespan_context') and hasattr(ctx.lifespan_context, 'db'):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, 'db'):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, 'db'):
        db = mcp.db
    
    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available for refresh")
        return {
            "error": "Database connection not available",
            "success": False
        }
    
    try:
        # Initialize response structure
        response = {
            "query": query,
            "success": True
        }
        
        # TIER 1: Load core identity information
        core_identity_node = os.environ.get("CORE_IDENTITY_NODE", "Alpha")
        logger.info(f"Loading core identity: {core_identity_node}")
        
        # Get the core identity with minimal depth (just the entity itself)
        core_identity = await db.get_entity(core_identity_node, depth=0)
        
        if core_identity:
            response["core_identity"] = core_identity
        else:
            logger.warning(f"Core identity node '{core_identity_node}' not found")
            # Continue anyway, as we might still get semantic results
        
        # TIER 2: Load contextually relevant memories based on the query
        relevant_memories = []
        
        # Check if db supports semantic search
        if hasattr(db, 'semantic_search'):
            # Limit the query to the first 1,000 characters as specified in ADR-008
            truncated_query = query[:1000] if len(query) > 1000 else query
            
            # Get the top 10 most relevant results as specified in ADR-008
            semantic_limit = 10
            
            logger.info(f"Performing semantic search with truncated query: '{truncated_query[:50]}...'")
            relevant_memories = await db.semantic_search(query=truncated_query, limit=semantic_limit)
            
            # Multi-level fallback logic as specified in ADR-008
            if len(relevant_memories) < 3:
                logger.info("Few semantic results, supplementing with recent observations")
                # TODO: Implement retrieval of recent observations
                # This would require a new database method
            
            if len(query) < 50:
                logger.info("Short query, using default set of important memories")
                # TODO: Implement retrieval of important memories
                # This would require a new database method
            
            response["relevant_memories"] = relevant_memories
        else:
            logger.warning("Database does not support semantic search for contextual memories")
            # Fall back to current approach - get the entity with relationships
            fallback_entity = await db.get_entity(core_identity_node, depth=1)
            if fallback_entity:
                response["fallback_entity"] = fallback_entity
        
        # Check if we have any useful information to return
        if not core_identity and not relevant_memories and "fallback_entity" not in response:
            logger.error("No information available for refresh operation")
            return {
                "query": query,
                "error": "No information available for refresh operation",
                "success": False
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in refresh: {str(e)}")
        return {
            "query": query,
            "error": f"Error in refresh: {str(e)}",
            "success": False
        }


@mcp.tool(name="remember")
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
    
    # Try to get the database connection from various places
    db = None
    
    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, 'lifespan_context') and hasattr(ctx.lifespan_context, 'db'):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, 'db'):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, 'db'):
        db = mcp.db
    
    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available")
        return {
            "error": "Database connection not available",
            "success": False
        }
    
    try:
        # Create or update the entity
        entity_result = await db.create_entity(entity, entity_type)
        
        # Add observation if provided
        if observation:
            observation_result = await db.add_observation(entity, observation)
            if not observation_result.get("success", False):
                # If adding observation failed, return the error
                return observation_result
        
        # Return the result
        return {
            "entity": entity,
            "type": entity_type or "Entity",
            "observation": observation,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error creating/updating entity: {str(e)}")
        return {
            "error": f"Error creating/updating entity: {str(e)}",
            "success": False
        }


@mcp.tool(name="relate")
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
    
    # Try to get the database connection from various places
    db = None
    
    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, 'lifespan_context') and hasattr(ctx.lifespan_context, 'db'):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, 'db'):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, 'db'):
        db = mcp.db
    
    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available")
        return {
            "error": "Database connection not available",
            "success": False
        }
    
    try:
        # Create the relationship
        result = await db.create_relationship(entity, to_entity, as_type)
        
        # Return the result
        return {
            "entity": entity,
            "to_entity": to_entity,
            "as_type": as_type,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error creating relationship: {str(e)}")
        return {
            "error": f"Error creating relationship: {str(e)}",
            "success": False
        }


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

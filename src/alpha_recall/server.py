"""
MCP server implementation for alpha_recall.

This module implements the Model Context Protocol (MCP) server for alpha_recall,
providing a structured memory system through Neo4j with semantic search capabilities
via Qdrant and sentence-transformers.
"""

import asyncio
import datetime
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import pytz
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP

from alpha_recall.db import GraphDatabase, create_db_instance
from alpha_recall.logging_utils import configure_logging, get_logger
from alpha_recall.models.entities import Entity, Observation, Relationship
from alpha_recall.utils.retry import async_retry
from alpha_recall.utils.alpha_snooze import create_alpha_snooze_from_env
from alpha_recall.reminiscer import ReminiscerAgent

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
        
        # Initialize reminiscer if enabled
        reminiscer = None
        reminiscer_enabled = os.environ.get("REMINISCER_ENABLED", "false").lower() == "true"
        logger.info(f"Reminiscer enabled check: REMINISCER_ENABLED='{os.environ.get('REMINISCER_ENABLED', 'NOT_SET')}', reminiscer_enabled={reminiscer_enabled}")
        if reminiscer_enabled:
            try:
                model_name = os.environ.get("REMINISCER_MODEL", "llama3.1:8b")
                ollama_host = os.environ.get("REMINISCER_OLLAMA_HOST", "localhost")
                ollama_port = int(os.environ.get("REMINISCER_OLLAMA_PORT", "11434"))
                
                reminiscer = ReminiscerAgent(
                    composite_db=db,
                    model_name=model_name,
                    ollama_host=ollama_host,
                    ollama_port=ollama_port
                )
                logger.info(f"Reminiscer initialized with model {model_name} at {ollama_host}:{ollama_port}")
            except Exception as e:
                logger.warning(f"Failed to initialize reminiscer: {e}")
                reminiscer = None

        # Yield the lifespan context with the database connection and reminiscer
        # Also set db directly on the mcp_server for tools that access ctx.db
        context = {"db": db, "reminiscer": reminiscer}
        mcp_server.db = db
        mcp_server.reminiscer = reminiscer
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

    @mcp.tool(name="recency_search")
    async def recency_search(ctx: Context, limit: int = 10) -> Dict[str, Any]:
        """
        Return the N most recent observations.
        Args:
            ctx: The request context containing lifespan resources
            limit: Maximum number of results to return (default 10)
        Returns:
            Dictionary containing the most recent observations
        """
        logger.info(f"Recency search called: limit={limit}")
        db = None
        if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
            db = ctx.lifespan_context.db
        elif hasattr(ctx, "db"):
            db = ctx.db
        elif hasattr(mcp, "db"):
            db = mcp.db
        if db is None:
            logger.error("Database connection not available for recency_search")
            return {"error": "Database connection not available", "success": False}
        if hasattr(db, "recency_search"):
            try:
                results = await db.recency_search(limit=limit)
                return {"results": results, "success": True}
            except Exception as e:
                logger.error(f"Error in recency_search: {str(e)}")
                return {"error": f"Error in recency_search: {str(e)}", "success": False}
        else:
            logger.error("recency_search not implemented in database backend")
            return {
                "error": "recency_search not implemented in database backend",
                "success": False,
            }

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
        if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
            db = ctx.lifespan_context.db
        elif hasattr(ctx, "db"):
            db = ctx.db
        elif hasattr(mcp, "db"):
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



@mcp.tool(name="search_shortterm")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def search_shortterm(
    ctx: Context, 
    query: str, 
    limit: int = 10,
    search_type: str = "semantic",
    through_the_last: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search short-term memories using semantic or emotional similarity.

    Args:
        ctx: The request context containing lifespan resources
        query: The natural language query to search for
        limit: Maximum number of results to return (default 10)
        search_type: Type of search - "semantic" or "emotional" (default "semantic")
        through_the_last: Optional time window (e.g., "24 hours", "1 week")

    Returns:
        Dictionary containing the short-term memory search results
    """
    logger.info(
        f"Search shortterm tool called: query='{query}', limit={limit}, search_type='{search_type}', through_the_last='{through_the_last}'"
    )

    # Try to get the database connection from various places
    db = None
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    elif hasattr(ctx, "db"):
        db = ctx.db
    elif hasattr(mcp, "db"):
        db = mcp.db

    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available for search_shortterm")
        return {"error": "Database connection not available", "success": False}

    try:
        # Validate search_type
        if search_type not in ["semantic", "emotional"]:
            return {
                "error": "search_type must be 'semantic' or 'emotional'",
                "success": False,
            }

        # Use the appropriate search method
        if search_type == "semantic":
            if hasattr(db, "semantic_search_shortterm"):
                results = await db.semantic_search_shortterm(
                    query=query, 
                    limit=limit, 
                    through_the_last=through_the_last
                )
            else:
                logger.error("Database does not support semantic search on short-term memory")
                return {
                    "error": "Database does not support semantic search on short-term memory",
                    "success": False,
                }
        else:  # emotional
            if hasattr(db, "emotional_search_shortterm"):
                results = await db.emotional_search_shortterm(
                    query=query, 
                    limit=limit, 
                    through_the_last=through_the_last
                )
            else:
                logger.error("Database does not support emotional search on short-term memory")
                return {
                    "error": "Database does not support emotional search on short-term memory",
                    "success": False,
                }

        return {
            "query": query,
            "search_type": search_type,
            "through_the_last": through_the_last,
            "results": results,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error performing short-term memory search: {str(e)}")
        return {
            "error": f"Error performing short-term memory search: {str(e)}",
            "success": False,
        }


@mcp.tool(name="search_longterm")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def search_longterm(
    ctx: Context,
    query: str,
    limit: int = 10,
    entity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search long-term memory observations using semantic similarity.

    Args:
        ctx: The request context containing lifespan resources
        query: The natural language query to search for
        limit: Maximum number of results to return (default 10)
        entity: Optional entity name to filter results

    Returns:
        Dictionary containing the long-term memory search results
    """
    logger.info(
        f"Search longterm tool called: query='{query}', limit={limit}, entity='{entity}'"
    )

    # Try to get the database connection from various places
    db = None
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    elif hasattr(ctx, "db"):
        db = ctx.db
    elif hasattr(mcp, "db"):
        db = mcp.db

    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available for search_longterm")
        return {"error": "Database connection not available", "success": False}

    try:
        # Check if db has semantic_search method (CompositeDatabase)
        if hasattr(db, "semantic_search"):
            results = await db.semantic_search(
                query=query, limit=limit, entity_name=entity
            )
            return {
                "query": query,
                "entity": entity,
                "results": results,
                "success": True,
            }
        else:
            logger.error("Database does not support long-term semantic search")
            return {
                "error": "Database does not support long-term semantic search",
                "success": False,
            }
    except Exception as e:
        logger.error(f"Error performing long-term memory search: {str(e)}")
        return {
            "error": f"Error performing long-term memory search: {str(e)}",
            "success": False,
        }


async def _get_time() -> Dict[str, Any]:
    """
    Returns the current time information in various formats.

    Returns:
        Dict containing ISO datetime, human-readable time, timezone info,
        day of week, and unix timestamp.
    """
    # Get the current time in Pacific timezone
    pacific_tz = pytz.timezone("America/Los_Angeles")
    now = datetime.datetime.now(pacific_tz)

    # Format for human-readable
    human_readable = now.strftime("%A, %B %d, %Y %I:%M %p")

    # Get day of week (1 = Monday, 7 = Sunday according to ISO)
    day_of_week_num = now.isoweekday()
    day_of_week_name = now.strftime("%A")

    # Get Unix timestamp
    unix_timestamp = int(time.time())

    # Build and return the response
    return {
        "iso_datetime": now.isoformat(),
        "human_readable": human_readable,
        "timezone": {
            "name": "America/Los_Angeles",
            "offset": now.strftime("%z"),
            "display": "Pacific Time",
        },
        "day_of_week": {"integer": day_of_week_num, "name": day_of_week_name},
        "unix_timestamp": unix_timestamp,
    }


@mcp.tool(name="refresh")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def refresh(ctx: Context, query: str) -> Dict[str, Any]:
    """
    Implements the enhanced bootstrap process as described in ADR-008 and ADR-009.

    This function loads:
    1. Tier 1: Core identity information (essential bootstrap entity)
    2. Tier 2: Recent short-term memories (from Redis)
    3. Tier 3: Contextually relevant memories based on the user's greeting

    Args:
        ctx: The request context containing lifespan resources
        query: The user's greeting or first message (used for semantic search)

    Returns:
        Dictionary containing:
        - core_identity: The core identity entity (Tier 1)
        - shortterm_memories: Recent short-term memories (Tier 2)
        - relevant_memories: Semantically relevant observations based on the query (Tier 3)
    """
    logger.info(f"Refresh tool called with query: '{query}'")

    # Try to get the database connection from various places
    db = None

    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, "db"):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, "db"):
        db = mcp.db

    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available for refresh")
        return {"error": "Database connection not available", "success": False}

    try:
        # Initialize response structure
        response = {"success": True}

        response["time"] = await _get_time()

        # TIER 1: Load core identity information
        core_identity_node = os.environ.get("CORE_IDENTITY_NODE", "Alpha")
        logger.info(f"Loading core identity: {core_identity_node}")

        # Get the core identity with depth=1 to include observations and relationships
        core_identity = await db.get_entity(core_identity_node, depth=1)

        if core_identity:
            # Only include the desired fields in the response
            allowed_keys = ("name", "updated_at", "observations", "relationships")
            if isinstance(core_identity, dict):
                filtered_core_identity = {
                    k: v for k, v in core_identity.items() if k in allowed_keys
                }
            else:
                filtered_core_identity = {
                    k: v for k, v in core_identity.dict().items() if k in allowed_keys
                }
            response["core_identity"] = filtered_core_identity
        else:
            logger.warning(f"Core identity node '{core_identity_node}' not found")
            # Continue anyway, as we might still get semantic results

        # TIER 2: Load recent short-term memories
        shortterm_memories = []
        if hasattr(db, "get_shortterm_memories"):
            try:
                # Get the 5 most recent short-term memories
                shortterm_limit = 5
                logger.info(f"Retrieving {shortterm_limit} recent short-term memories")
                shortterm_memories = await db.get_shortterm_memories(
                    limit=shortterm_limit
                )
                logger.info(f"Retrieved {len(shortterm_memories)} short-term memories")

                # Filter short-term memories to only include essential fields
                filtered_shortterm = [
                    {
                        "content": memory.get("content"),
                        "created_at": memory.get("created_at"),
                        "client": memory.get("client", {}),
                    }
                    for memory in shortterm_memories
                ]

                # Add filtered short-term memories to the response
                response["shortterm_memories"] = filtered_shortterm
            except Exception as e:
                logger.error(
                    f"Error retrieving short-term memories during refresh: {str(e)}"
                )
                # Continue with other retrievals even if short-term memory fails
        else:
            logger.info("Short-term memory retrieval not supported in the database")

        # TIER 3: Load contextually relevant memories based on the query
        relevant_memories = []
        recent_observations = []

        # Check if db supports semantic search
        if hasattr(db, "semantic_search"):
            # Limit the query to the first 1,000 characters as specified in ADR-008
            truncated_query = query[:1000] if len(query) > 1000 else query

            # Get the top 10 most relevant results as specified in ADR-008
            semantic_limit = 10

            logger.info(
                f"Performing semantic search with truncated query: '{truncated_query[:50]}...'"
            )
            relevant_memories = await db.semantic_search(
                query=truncated_query, limit=semantic_limit
            )

            # Also get the most recent N observations
            if hasattr(db, "recency_search"):
                try:
                    recent_observations = await db.recency_search(limit=semantic_limit)
                except Exception as e:
                    logger.error(f"Error in recency_search during refresh: {str(e)}")
                    recent_observations = []
            else:
                logger.info("recency_search not implemented in db for refresh")

            # Multi-level fallback logic as specified in ADR-008
            if len(relevant_memories) < 3:
                logger.info(
                    "Few semantic results, supplementing with recent observations"
                )
                # TODO: Implement retrieval of recent observations
                # This would require a new database method

            if len(query) < 50:
                logger.info("Short query, using default set of important memories")
                # TODO: Implement retrieval of important memories
                # This would require a new database method

            response["relevant_memories"] = relevant_memories
            # Only include 'created_at' and 'content' in each recent observation
            filtered_recent = [
                {
                    "created_at": obs.get("created_at"),
                    "entity_name": (
                        obs.get("entity_name")
                        if "entity_name" in obs
                        else obs.get("entity") if "entity" in obs else None
                    ),
                    "content": obs.get("content"),
                }
                for obs in recent_observations
            ]
            response["recent_observations"] = filtered_recent
        else:
            logger.warning(
                "Database does not support semantic search for contextual memories"
            )
            # Fall back to current approach - get the entity with relationships
            fallback_entity = await db.get_entity(core_identity_node, depth=1)
            if fallback_entity:
                response["fallback_entity"] = fallback_entity

        # Check if we have any useful information to return
        if (
            not core_identity
            and not relevant_memories
            and "fallback_entity" not in response
        ):
            logger.error("No information available for refresh operation")
            return {
                "error": "No information available for refresh operation",
                "success": False,
            }

        return response

    except Exception as e:
        logger.error(f"Error in refresh: {str(e)}")
        return {"error": f"Error in refresh: {str(e)}", "success": False}


@mcp.tool(name="gentle_refresh")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def gentle_refresh(ctx: Context, query: Optional[str] = None) -> Dict[str, Any]:
    """
    Simplified refresh function focused on temporal orientation rather than semantic search.
    
    Designed to solve memory orientation problems by providing:
    1. Current time information for temporal grounding
    2. Core identity observations (natural language facts, not relationship triples)
    3. 10 most recent short-term memories for contextual orientation
    4. Alpha-Snooze memory consolidation (if enabled) - processes recent memories for insights
    5. 5 most recent observations for slow-changing facts
    
    Eliminates cognitive overload from semantic search and prioritizes temporal
    orientation over semantic relevance. When Alpha-Snooze is enabled, provides
    additional memory consolidation insights extracted from recent interactions.
    
    Args:
        ctx: The request context containing lifespan resources
        query: Optional query parameter (accepted for compatibility but ignored)
        
    Returns:
        Dictionary containing:
        - time: Current time information
        - core_identity: Essential identity observations (observations only, no relationships)
        - shortterm_memories: 10 most recent short-term memories
        - memory_consolidation: Alpha-Snooze insights (if enabled and available)
        - recent_observations: 5 most recent observations
    """
    logger.info("Gentle refresh tool called")
    
    # Try to get the database connection from various places
    db = None
    
    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, "db"):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, "db"):
        db = mcp.db
    
    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available for gentle_refresh")
        return {"error": "Database connection not available", "success": False}
    
    try:
        # Initialize response structure
        response = {"success": True}
        
        # Get current time for temporal orientation
        response["time"] = await _get_time()
        
        # Core identity: Get observations only (no relationship triples)
        core_identity_node = os.environ.get("CORE_IDENTITY_NODE", "Alpha")
        logger.info(f"Loading core identity: {core_identity_node}")
        
        core_identity = await db.get_entity(core_identity_node, depth=1)
        
        if core_identity:
            # Only include observations, skip relationships to avoid analysis mode
            core_identity_filtered = {
                "name": core_identity.get("name") if isinstance(core_identity, dict) else core_identity.name,
                "updated_at": core_identity.get("updated_at") if isinstance(core_identity, dict) else core_identity.updated_at,
                "observations": core_identity.get("observations") if isinstance(core_identity, dict) else core_identity.observations,
                # Deliberately exclude relationships to focus on identity facts
            }
            response["core_identity"] = core_identity_filtered
        else:
            logger.warning(f"Core identity node '{core_identity_node}' not found")
        
        # Short-term memories: Get 10 most recent for temporal orientation
        shortterm_memories = []
        if hasattr(db, "get_shortterm_memories"):
            try:
                shortterm_limit = 10  # Increased from 5 for better temporal context
                logger.info(f"Retrieving {shortterm_limit} recent short-term memories")
                shortterm_memories = await db.get_shortterm_memories(limit=shortterm_limit)
                logger.info(f"Retrieved {len(shortterm_memories)} short-term memories")
                
                # Filter short-term memories to essential fields
                filtered_shortterm = [
                    {
                        "content": memory.get("content"),
                        "created_at": memory.get("created_at"),
                        "client": memory.get("client", {}),
                    }
                    for memory in shortterm_memories
                ]
                
                response["shortterm_memories"] = filtered_shortterm
                
                # Alpha-Snooze: Memory consolidation (if enabled and available)
                try:
                    alpha_snooze = await create_alpha_snooze_from_env()
                    if alpha_snooze:
                        # Get memories for alpha-snooze time window (may be different from gentle_refresh limit)
                        snooze_memories = await db.get_shortterm_memories(
                            through_the_last=alpha_snooze.time_window,
                            limit=100  # High limit since we're using time window
                        )
                        if snooze_memories:
                            logger.info(f"Running alpha-snooze memory consolidation on {len(snooze_memories)} memories from last {alpha_snooze.time_window}")
                            consolidation = await alpha_snooze.consolidate_memories(snooze_memories)
                            if consolidation:
                                response["memory_consolidation"] = consolidation
                                logger.info(f"Alpha-snooze processed {consolidation.get('processed_memories_count', 0)} memories")
                            else:
                                logger.info("Alpha-snooze consolidation returned no results")
                        else:
                            logger.info(f"Alpha-snooze available but no short-term memories in last {alpha_snooze.time_window}")
                    else:
                        logger.info("Alpha-snooze not available for memory consolidation")
                except Exception as e:
                    logger.warning(f"Alpha-snooze consolidation failed: {str(e)}")
                    # Continue with gentle_refresh even if alpha-snooze fails
                
            except Exception as e:
                logger.error(f"Error retrieving short-term memories during gentle_refresh: {str(e)}")
                # Continue with other retrievals even if short-term memory fails
        else:
            logger.info("Short-term memory retrieval not supported in the database")
        
        # Recent observations: Get 5 most recent for slow-changing facts
        recent_observations = []
        if hasattr(db, "recency_search"):
            try:
                recent_limit = 5
                logger.info(f"Retrieving {recent_limit} most recent observations")
                recent_observations = await db.recency_search(limit=recent_limit)
                logger.info(f"Retrieved {len(recent_observations)} recent observations")
                
                # Filter to essential fields
                filtered_recent = [
                    {
                        "created_at": obs.get("created_at"),
                        "entity_name": (
                            obs.get("entity_name")
                            if "entity_name" in obs
                            else obs.get("entity") if "entity" in obs else None
                        ),
                        "content": obs.get("content"),
                    }
                    for obs in recent_observations
                ]
                response["recent_observations"] = filtered_recent
            except Exception as e:
                logger.error(f"Error retrieving recent observations during gentle_refresh: {str(e)}")
                response["recent_observations"] = []
        else:
            logger.info("recency_search not available in database")
            response["recent_observations"] = []
        
        return response
        
    except Exception as e:
        logger.error(f"Error in gentle_refresh: {str(e)}")
        return {"error": f"Error in gentle_refresh: {str(e)}", "success": False}


@mcp.tool(name="remember_longterm")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def remember_longterm(
    ctx: Context,
    entity: str,
    entity_type: Optional[str] = None,
    observation: Optional[str] = None,
) -> Dict[str, Any]:
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
    logger.info(
        f"Remember longterm tool called: entity='{entity}', type='{entity_type}', observation='{observation}'"
    )

    # Try to get the database connection from various places
    db = None

    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, "db"):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, "db"):
        db = mcp.db

    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available")
        return {"error": "Database connection not available", "success": False}

    try:
        # Create or update the entity
        entity_result = await db.create_entity(entity, entity_type)

        # Add observation if provided
        if observation:
            observation_result = await db.add_observation(entity, observation)
            # Only treat as error if success=False is explicitly set
            if (
                isinstance(observation_result, dict)
                and observation_result.get("success") is False
            ):
                return {
                    "success": False,
                    "message": observation_result.get(
                        "error", "Failed to add observation."
                    ),
                }
        # Always return success if no error above
        return {"success": True}

    except Exception as e:
        logger.error(f"Error creating/updating entity: {str(e)}")
        return {
            "success": False,
            "message": f"Error creating/updating entity: {str(e)}",
        }


@mcp.tool(name="remember_shortterm")
async def remember_shortterm(ctx: Context, content: str) -> Dict[str, Any]:
    """
    Store a short-term memory with automatic TTL expiration and return relevant memories.

    Args:
        ctx: The request context containing lifespan resources
        content: The memory content to store

    Returns:
        Dictionary containing information about the stored memory and 5 most relevant memories
    """
    logger.info(
        f"Remember shortterm tool called: content='{content[:50]}...' if len(content) > 50 else content"
    )

    # Try to get the database connection from various places
    db = None

    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, "db"):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, "db"):
        db = mcp.db

    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available")
        return {"error": "Database connection not available", "success": False}

    try:
        # Store the short-term memory (this now includes relevant memories)
        result = await db.remember_shortterm(content)
        
        # Build response
        response = {"success": result.get("success", True)}
        
        # If the result contains created_at, include it as timestamp
        if isinstance(result, dict) and "created_at" in result:
            response["timestamp"] = result["created_at"]
        
        # Include relevant memories if available
        if isinstance(result, dict) and "relevant_memories" in result:
            # Format relevant memories for cleaner output
            relevant_memories = []
            for memory in result["relevant_memories"]:
                formatted_memory = {
                    "content": memory.get("content"),
                    "created_at": memory.get("created_at"),
                    "search_type": memory.get("search_type"),
                    "score": memory.get("score", 0),
                }
                # Include client info if present
                if "client" in memory:
                    formatted_memory["client"] = memory["client"]
                relevant_memories.append(formatted_memory)
            
            response["relevant_memories"] = relevant_memories
            
        return response

    except Exception as e:
        logger.error(f"Error storing short-term memory: {str(e)}")
        return {"success": False, "error": f"Error storing short-term memory: {str(e)}"}


@mcp.tool(name="relate_longterm")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def relate_longterm(
    ctx: Context, entity: str, to_entity: str, as_type: str
) -> Dict[str, Any]:
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
    logger.info(
        f"Relate longterm tool called: entity='{entity}', to_entity='{to_entity}', as_type='{as_type}'"
    )

    # Try to get the database connection from various places
    db = None

    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, "db"):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, "db"):
        db = mcp.db

    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available")
        return {"error": "Database connection not available", "success": False}

    try:
        # Create the relationship
        result = await db.create_relationship(entity, to_entity, as_type)

        # Return the result
        return {
            "entity": entity,
            "to_entity": to_entity,
            "as_type": as_type,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error creating relationship: {str(e)}")
        return {"error": f"Error creating relationship: {str(e)}", "success": False}


@mcp.tool(name="remember_narrative")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def remember_narrative(
    ctx: Context,
    title: str,
    paragraphs: List[str],
    participants: List[str],
    tags: Optional[List[str]] = None,
    outcome: str = "ongoing",
    references: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Store a narrative memory with hybrid storage (Redis content + Memgraph relationships).
    
    Narrative memories capture the stories and context around conversations, decisions,
    and experiences. They use dual-granularity embeddings (story-level and paragraph-level)
    for flexible search and retrieval.

    Args:
        ctx: The request context containing lifespan resources
        title: Story title (e.g., "Debugging the Embedding Pipeline")
        paragraphs: List of paragraph texts that make up the story
        participants: List of participant names (e.g., ["Alpha", "Jeffery"])
        tags: Optional list of tags/topics (e.g., ["debugging", "breakthrough"])
        outcome: Story outcome - "breakthrough", "resolution", "ongoing", "blocked"
        references: Optional list of story_ids this story references or builds upon

    Returns:
        Dictionary containing the created story information and success status
    """
    logger.info(f"Remember narrative tool called: title='{title}', participants={participants}")
    
    # Fix: Parse JSON string parameters if they come in as strings
    import json
    
    if isinstance(paragraphs, str):
        try:
            paragraphs = json.loads(paragraphs)
            logger.info(f"Parsed paragraphs from JSON string to list of {len(paragraphs)} items")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse paragraphs JSON: {e}")
            return {"success": False, "error": f"Invalid paragraphs format: {e}"}
    
    if isinstance(participants, str):
        try:
            participants = json.loads(participants)
            logger.info(f"Parsed participants from JSON string to list of {len(participants)} items")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse participants JSON: {e}")
            return {"success": False, "error": f"Invalid participants format: {e}"}
    
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
            logger.info(f"Parsed tags from JSON string to list of {len(tags)} items")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tags JSON: {e}")
            return {"success": False, "error": f"Invalid tags format: {e}"}
    
    if isinstance(references, str):
        try:
            references = json.loads(references)
            logger.info(f"Parsed references from JSON string to list of {len(references)} items")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse references JSON: {e}")
            return {"success": False, "error": f"Invalid references format: {e}"}

    # Try to get the database connection from various places
    db = None

    # Method 1: Try to get from lifespan_context
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    # Method 2: Try to get directly from context
    elif hasattr(ctx, "db"):
        db = ctx.db
    # Method 3: Try to get from the MCP server
    elif hasattr(mcp, "db"):
        db = mcp.db

    # If we couldn't get a database connection, return a meaningful error
    if db is None:
        logger.error("Database connection not available")
        return {"error": "Database connection not available", "success": False}

    try:
        # Store the narrative memory
        result = await db.store_narrative(
            title=title,
            paragraphs=paragraphs,
            participants=participants,
            tags=tags,
            outcome=outcome,
            references=references,
        )

        if result.get("success", False):
            logger.info(f"Successfully stored narrative: {result.get('story_id')}")
            
            # Format the response with story details
            response = {
                "success": True,
                "story_id": result.get("story_id"),
                "title": title,
                "created_at": result.get("created_at"),
                "paragraph_count": result.get("paragraph_count", len(paragraphs)),
                "participants": participants,
                "tags": tags or [],
                "outcome": outcome,
                "message": f"Narrative '{title}' stored successfully with {len(paragraphs)} paragraphs and dual-granularity embeddings."
            }
            
            if references:
                response["references"] = references
                
            return response
        else:
            logger.error(f"Failed to store narrative: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown error storing narrative")
            }

    except Exception as e:
        logger.error(f"Error storing narrative memory: {str(e)}")
        return {"success": False, "error": f"Error storing narrative memory: {str(e)}"}


@mcp.tool(name="search_narratives")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def search_narratives(
    ctx: Context,
    query: str,
    search_type: str = "semantic",
    granularity: str = "story",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Search narrative memories using vector similarity.
    
    Supports multiple search modes:
    - Semantic: Find stories with similar meaning/topics
    - Emotional: Find stories with similar emotional resonance
    - Both: Combined semantic and emotional search
    
    And multiple granularities:
    - Story: Search at story level (find whole stories)
    - Paragraph: Search at paragraph level (find specific moments)
    - Both: Search both levels and merge results

    Args:
        ctx: The request context containing lifespan resources
        query: Search query (e.g., "debugging breakthrough moments")
        search_type: "semantic", "emotional", or "both"
        granularity: "story", "paragraph", or "both"
        limit: Maximum number of results to return

    Returns:
        Dictionary containing matching stories/paragraphs with similarity scores
    """
    logger.info(f"Search narratives tool called: query='{query}', type={search_type}, granularity={granularity}")

    # Try to get the database connection
    db = None
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    elif hasattr(ctx, "db"):
        db = ctx.db
    elif hasattr(mcp, "db"):
        db = mcp.db

    if db is None:
        logger.error("Database connection not available")
        return {"error": "Database connection not available", "success": False}

    try:
        # Perform the search
        results = await db.search_narratives(
            query=query,
            search_type=search_type,
            granularity=granularity,
            limit=limit,
        )

        return {
            "success": True,
            "query": query,
            "search_type": search_type,
            "granularity": granularity,
            "total_results": len(results),
            "results": results,
            "message": f"Found {len(results)} narrative matches for '{query}' using {search_type} search at {granularity} level."
        }

    except Exception as e:
        logger.error(f"Error searching narrative memories: {str(e)}")
        return {"success": False, "error": f"Error searching narrative memories: {str(e)}"}


@mcp.tool(name="recall_narrative")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def recall_narrative(
    ctx: Context,
    story_id: str,
) -> Dict[str, Any]:
    """
    Retrieve a complete narrative story by its story_id.
    
    Args:
        ctx: The request context containing lifespan resources
        story_id: The unique story identifier (e.g., "story_1719012345_a1b2c3d4")
    
    Returns:
        Dictionary containing the complete story with all paragraphs and metadata
    """
    logger.info(f"Recall narrative tool called: story_id='{story_id}'")

    # Try to get the database connection
    db = None
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    elif hasattr(ctx, "db"):
        db = ctx.db
    elif hasattr(mcp, "db"):
        db = mcp.db

    if db is None:
        logger.error("Database connection not available")
        return {"error": "Database connection not available", "success": False}

    try:
        # Retrieve the complete story
        story = await db.get_narrative(story_id)
        
        if story is None:
            return {
                "success": False,
                "error": f"Story with ID '{story_id}' not found",
                "story_id": story_id
            }

        return {
            "success": True,
            "story": story,
            "message": f"Retrieved story '{story.get('title', 'Untitled')}' with {len(story.get('paragraphs', []))} paragraphs."
        }

    except Exception as e:
        logger.error(f"Error retrieving narrative story: {str(e)}")
        return {"success": False, "error": f"Error retrieving narrative story: {str(e)}", "story_id": story_id}


@mcp.tool(name="list_narratives")
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"],
)
async def list_narratives(
    ctx: Context,
    limit: int = 10,
    offset: int = 0,
    since: str = None,
    participants: list = None,
    tags: list = None,
    outcome: str = None,
) -> Dict[str, Any]:
    """
    List narrative stories chronologically with optional filtering.
    
    This tool provides browsing functionality for narrative memories, allowing you to:
    - Browse stories chronologically (most recent first)
    - Filter by time window (e.g., "2d", "1w", "1m")
    - Filter by participants, tags, or outcome
    - Paginate through large result sets
    
    Args:
        ctx: The request context containing lifespan resources
        limit: Maximum number of stories to return (default: 10)
        offset: Number of stories to skip for pagination (default: 0)
        since: Time window filter (e.g., "2d", "1w", "1m", "6h") (optional)
        participants: Filter by participants list (AND logic) (optional)
        tags: Filter by tags list (AND logic) (optional)
        outcome: Filter by outcome ("breakthrough", "resolution", "ongoing", "blocked") (optional)

    Returns:
        Dictionary containing:
        - stories: List of story metadata (title, story_id, created_at, participants, tags, outcome)
        - total_count: Total number of stories matching filters
        - returned_count: Number of stories in this response
        - offset: Current pagination offset
        - limit: Requested limit
        - has_more: Whether there are more stories available
    """
    logger.info(f"List narratives tool called: limit={limit}, offset={offset}, since={since}")

    # Try to get the database connection
    db = None
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    elif hasattr(ctx, "db"):
        db = ctx.db
    elif hasattr(mcp, "db"):
        db = mcp.db

    if not db:
        logger.error("Database connection not available")
        return {
            "success": False,
            "error": "Database connection not available",
            "stories": [],
            "total_count": 0,
            "returned_count": 0
        }

    try:
        # Call the database method
        result = await db.list_narratives(
            limit=limit,
            offset=offset,
            since=since,
            participants=participants,
            tags=tags,
            outcome=outcome
        )

        # Add success status to the result
        result["success"] = True
        
        # Create a human-readable summary
        filter_parts = []
        if since:
            filter_parts.append(f"since {since}")
        if participants:
            filter_parts.append(f"involving {', '.join(participants)}")
        if tags:
            filter_parts.append(f"tagged with {', '.join(tags)}")
        if outcome:
            filter_parts.append(f"with outcome '{outcome}'")
        
        filter_desc = f" ({', '.join(filter_parts)})" if filter_parts else ""
        
        if "error" in result:
            result["message"] = f"Failed to list narratives{filter_desc}: {result['error']}"
        else:
            result["message"] = (
                f"Found {result['total_count']} narrative stories{filter_desc}. "
                f"Returning {result['returned_count']} stories "
                f"(offset {offset}, limit {limit})"
                f"{' - More available' if result.get('has_more') else ''}"
            )

        return result

    except Exception as e:
        logger.error(f"Error listing narrative stories: {str(e)}")
        return {
            "success": False,
            "error": f"Error listing narrative stories: {str(e)}",
            "stories": [],
            "total_count": 0,
            "returned_count": 0,
            "offset": offset,
            "limit": limit,
            "has_more": False
        }


@mcp.tool(name="search_all_memories")
async def search_all_memories(
    ctx: Context,
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Search across all memory systems (STM, LTM, NM) with unified results.
    
    This is a unified memory search tool that searches across all three memory subsystems:
    - Short-term memories (Redis with TTL) - semantic and emotional search
    - Long-term observations (Qdrant vector store) - semantic search
    - Narrative memories (Redis with embeddings) - semantic and emotional search
    - Entity names (exact matching)
    
    Results are merged and sorted by similarity score (cosine distance) to provide
    serendipitous memory discovery across all storage systems.
    
    Args:
        ctx: The request context containing lifespan resources
        query: Search query to find relevant memories
        limit: Maximum number of results to return (default: 10)
        offset: Number of results to skip for pagination (default: 0)
    
    Returns:
        Dictionary containing unified search results from all memory systems
    """
    logger.info(f"Search all memories tool called: query='{query}', limit={limit}, offset={offset}")
    
    # Get database connection
    db = None
    if hasattr(ctx, "lifespan_context") and hasattr(ctx.lifespan_context, "db"):
        db = ctx.lifespan_context.db
    elif hasattr(ctx, "db"):
        db = ctx.db
    elif hasattr(mcp, "db"):
        db = mcp.db

    if db is None:
        logger.error("Database connection not available")
        return {"error": "Database connection not available", "success": False}

    try:
        all_results = []
        
        # Search STM (short-term memory) - both semantic and emotional
        try:
            stm_semantic = await db.semantic_search_shortterm(query, limit=50)
            for result in stm_semantic:
                all_results.append({
                    "source": "STM",
                    "search_type": "semantic", 
                    "content": result.get("content", ""),
                    "score": result.get("similarity_score", 0.0),
                    "created_at": result.get("created_at"),
                    "client": result.get("client", {}),
                    "id": f"stm_{result.get('id', hash(result.get('content', '')))}"
                })
        except Exception as e:
            logger.warning(f"STM semantic search failed: {e}")
        
        try:
            stm_emotional = await db.emotional_search_shortterm(query, limit=50)
            for result in stm_emotional:
                all_results.append({
                    "source": "STM",
                    "search_type": "emotional",
                    "content": result.get("content", ""),
                    "score": result.get("emotional_score", 0.0),
                    "created_at": result.get("created_at"),
                    "client": result.get("client", {}),
                    "id": f"stm_{result.get('id', hash(result.get('content', '')))}"
                })
        except Exception as e:
            logger.warning(f"STM emotional search failed: {e}")
        
        # Search LTM (long-term memory observations)
        try:
            ltm_results = await db.semantic_search(query, limit=50)
            for result in ltm_results:
                all_results.append({
                    "source": "LTM",
                    "search_type": "semantic",
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "created_at": result.get("created_at"),
                    "entity_name": result.get("entity_name"),
                    "id": f"ltm_{result.get('id', hash(result.get('content', '')))}"
                })
        except Exception as e:
            logger.warning(f"LTM semantic search failed: {e}")
        
        # Search NM (narrative memory) - both semantic and emotional
        try:
            nm_semantic = await db.search_narratives(query, search_type="semantic", granularity="both", limit=25)
            for result in nm_semantic:
                all_results.append({
                    "source": "NM",
                    "search_type": "semantic",
                    "content": result.get("content", result.get("title", "")),
                    "score": result.get("score", 0.0),
                    "created_at": result.get("created_at"),
                    "story_id": result.get("story_id"),
                    "title": result.get("title"),
                    "granularity": result.get("granularity"),
                    "id": f"nm_{result.get('story_id', hash(result.get('content', '')))}"
                })
        except Exception as e:
            logger.warning(f"NM semantic search failed: {e}")
        
        try:
            nm_emotional = await db.search_narratives(query, search_type="emotional", granularity="both", limit=25)
            for result in nm_emotional:
                all_results.append({
                    "source": "NM",
                    "search_type": "emotional",
                    "content": result.get("content", result.get("title", "")),
                    "score": result.get("score", 0.0),
                    "created_at": result.get("created_at"),
                    "story_id": result.get("story_id"),
                    "title": result.get("title"),
                    "granularity": result.get("granularity"),
                    "id": f"nm_{result.get('story_id', hash(result.get('content', '')))}"
                })
        except Exception as e:
            logger.warning(f"NM emotional search failed: {e}")
        
        # Search entities (exact name matching)
        try:
            entity_result = await db.get_entity(query, depth=1)
            if entity_result:
                all_results.append({
                    "source": "ENTITY",
                    "search_type": "exact_match",
                    "content": f"Entity: {query}",
                    "score": 1.0,  # Perfect match for exact entity name
                    "entity_name": query,
                    "entity_data": entity_result,
                    "id": f"entity_{hash(query)}"
                })
        except Exception as e:
            logger.warning(f"Entity search failed: {e}")
        
        # Sort all results by score (descending - higher scores are more relevant)
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Remove duplicates by ID (keeping highest scored version)
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = result.get("id")
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        # Apply pagination
        total_found = len(unique_results)
        paginated_results = unique_results[offset:offset + limit]
        
        return {
            "success": True,
            "query": query,
            "results": paginated_results,
            "total_found": total_found,
            "returned_count": len(paginated_results),
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total_found,
            "sources_searched": ["STM", "LTM", "NM", "ENTITIES"],
            "message": f"Found {total_found} memories across all systems for '{query}'. Returning {len(paginated_results)} results (offset {offset})."
        }
        
    except Exception as e:
        logger.error(f"Error in search_all_memories: {str(e)}")
        return {"success": False, "error": f"Error searching all memories: {str(e)}"}


@mcp.tool(name="ask_memory")
async def ask_memory(ctx: Context, question: str, new_chat: bool = False) -> Dict[str, Any]:
    """
    Ask a conversational question to Alpha-Reminiscer about memories.
    
    This tool provides a natural language interface to Alpha's memory systems,
    allowing for complex questions and maintaining conversation context.
    
    Args:
        ctx: The request context containing lifespan resources
        question: Natural language question about memories
        new_chat: If True, reset conversation context and start fresh (default: False)
    
    Returns:
        Dictionary containing the reminiscer's response
    """
    logger.info(f"Ask memory tool called: question='{question}'")
    logger.debug(f"[ASK_MEMORY] Question received: {question}")
    
    # Get reminiscer from context
    reminiscer = None
    if hasattr(ctx, "lifespan_context") and ctx.lifespan_context.get("reminiscer"):
        reminiscer = ctx.lifespan_context["reminiscer"]
    elif hasattr(ctx, "reminiscer"):
        reminiscer = ctx.reminiscer
    elif hasattr(mcp, "reminiscer"):
        reminiscer = mcp.reminiscer
    
    if reminiscer is None:
        return {
            "success": False,
            "error": "Alpha-Reminiscer is not available. Set REMINISCER_ENABLED=true and ensure Ollama is running.",
            "question": question
        }
    
    try:
        # Reset conversation if requested
        if new_chat:
            reminiscer.reset_conversation()
            logger.debug(f"[ASK_MEMORY] Conversation context reset for new chat")
        
        response = await reminiscer.ask(question)
        conversation_length = reminiscer.get_conversation_length()
        
        logger.debug(f"[ASK_MEMORY] Response from reminiscer: {response}")
        logger.debug(f"[ASK_MEMORY] Conversation length: {conversation_length}")
        
        return {
            "success": True,
            "question": question,
            "response": response,
            "conversation_length": conversation_length,
            "new_chat": new_chat,
            "message": f"Alpha-Reminiscer processed your question about memories."
        }
        
    except Exception as e:
        logger.error(f"Error in ask_memory: {str(e)}")
        return {
            "success": False,
            "error": f"Error asking reminiscer: {str(e)}",
            "question": question
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

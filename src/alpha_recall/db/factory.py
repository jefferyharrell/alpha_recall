"""
Factory for creating database instances.

This module provides factory functions for creating database instances,
including graph databases, vector stores, and composite databases.
"""

import os
from typing import Optional, Type, Union

from dotenv import load_dotenv

from alpha_recall.db.base import GraphDatabase
from alpha_recall.db.composite_db import CompositeDatabase
from alpha_recall.db.memgraph_db import MemgraphDatabase
from alpha_recall.db.neo4j_db import Neo4jDatabase
from alpha_recall.db.redis_db import RedisShortTermMemory
from alpha_recall.db.semantic_search import SemanticSearch
from alpha_recall.db.vector_store import VectorStore
from alpha_recall.db.memgraph_vector_store import MemgraphVectorStore
from alpha_recall.db.narrative_memory import NarrativeMemory
from alpha_recall.logging_utils import get_logger

# Load environment variables
load_dotenv()

# Get logger
logger = get_logger("db_factory")

# Database configuration from environment
GRAPH_DB_TYPE = os.environ.get("GRAPH_DB", "memgraph").lower()
GRAPH_DB_URI = os.environ.get("GRAPH_DB_URI", "bolt://localhost:7687")
VECTOR_STORE_TYPE = os.environ.get("VECTOR_STORE_TYPE", "memgraph").lower()  # "qdrant" or "memgraph"
VECTOR_STORE_URL = os.environ.get("VECTOR_STORE_URL", "http://localhost:6333")
VECTOR_STORE_COLLECTION = os.environ.get(
    "VECTOR_STORE_COLLECTION", "alpha_recall_observations_768d"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMOTIONAL_EMBEDDING_MODEL = os.environ.get("EMOTIONAL_EMBEDDING_MODEL", "j-hartmann/emotion-english-distilroberta-base")

# Redis configuration from environment
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_TTL = int(os.environ.get("REDIS_TTL", 259200))  # Default: 72 hours


async def create_graph_db() -> GraphDatabase:
    """
    Create and connect to a graph database instance based on environment configuration.

    Returns:
        Connected GraphDatabase instance

    Raises:
        ValueError: If the configured database type is not supported
    """
    if GRAPH_DB_TYPE == "neo4j":
        logger.info("Creating Neo4j database instance")
        db = Neo4jDatabase()
    elif GRAPH_DB_TYPE == "memgraph":
        logger.info("Creating Memgraph database instance")
        db = MemgraphDatabase()
    else:
        logger.error(f"Unsupported database type: {GRAPH_DB_TYPE}")
        raise ValueError(f"Unsupported database type: {GRAPH_DB_TYPE}")

    # Connect to the database
    await db.connect()
    return db


def create_vector_store() -> SemanticSearch:
    """
    Create a vector store instance for semantic search.

    Returns:
        SemanticSearch instance
    """
    if VECTOR_STORE_TYPE == "qdrant":
        logger.info(f"Creating Qdrant vector store with URL: {VECTOR_STORE_URL}")
        return VectorStore(
            qdrant_url=VECTOR_STORE_URL,
            collection_name=VECTOR_STORE_COLLECTION,
            model_name=EMBEDDING_MODEL,
        )
    elif VECTOR_STORE_TYPE == "memgraph":
        logger.info(f"Creating Memgraph vector store with URI: {GRAPH_DB_URI}")
        return MemgraphVectorStore(
            memgraph_uri=GRAPH_DB_URI,
            model_name=EMBEDDING_MODEL,
        )
    else:
        logger.error(f"Unsupported vector store type: {VECTOR_STORE_TYPE}")
        raise ValueError(f"Unsupported vector store type: {VECTOR_STORE_TYPE}")


async def create_shortterm_memory() -> Optional[RedisShortTermMemory]:
    """
    Create a Redis-based short-term memory instance.

    Returns:
        RedisShortTermMemory instance or None if creation fails
    """
    try:
        logger.info(
            f"Creating Redis short-term memory with host: {REDIS_HOST}:{REDIS_PORT}"
        )
        stm = RedisShortTermMemory(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD if REDIS_PASSWORD else None,
            db=REDIS_DB,
            ttl=REDIS_TTL,
            embedding_model=EMBEDDING_MODEL,
            emotional_embedding_model=EMOTIONAL_EMBEDDING_MODEL,
        )

        # Connect to Redis
        await stm.connect()

        if await stm.is_connected():
            logger.info("Successfully connected to Redis for short-term memory")
            return stm
        else:
            logger.warning("Failed to connect to Redis for short-term memory")
            return None
    except Exception as e:
        logger.error(f"Error creating Redis short-term memory: {str(e)}")
        return None


async def create_narrative_memory(
    graph_db: GraphDatabase, 
    shortterm_memory: Optional[RedisShortTermMemory]
) -> Optional[NarrativeMemory]:
    """
    Create a narrative memory instance if enabled and Redis is available.

    Args:
        graph_db: Connected graph database instance
        shortterm_memory: Connected Redis short-term memory instance

    Returns:
        NarrativeMemory instance or None if unavailable
    """
    if not shortterm_memory:
        logger.warning("Cannot create narrative memory without Redis connection")
        return None

    try:
        logger.info("Creating narrative memory instance")
        return NarrativeMemory(
            redis_client=shortterm_memory.client,
            graph_db=graph_db,
            embedding_model=EMBEDDING_MODEL,
            emotional_embedding_model=EMOTIONAL_EMBEDDING_MODEL,
        )
    except Exception as e:
        logger.error(f"Error creating narrative memory: {str(e)}")
        return None


async def create_db_instance() -> Union[GraphDatabase, CompositeDatabase]:
    """
    Create a composite database instance with graph database, vector store, and short-term memory.

    Returns:
        CompositeDatabase instance that combines all database components

    Raises:
        ValueError: If the configured database type is not supported
    """
    # Create graph database
    graph_db = await create_graph_db()

    # Create vector store
    vector_store = create_vector_store()

    # Create short-term memory store
    shortterm_memory = await create_shortterm_memory()

    # Create narrative memory (if enabled)
    narrative_memory = await create_narrative_memory(graph_db, shortterm_memory)

    # Create composite database
    logger.info("Creating composite database instance")
    return CompositeDatabase(
        graph_db=graph_db,
        semantic_search=vector_store,
        shortterm_memory=shortterm_memory,
        narrative_memory=narrative_memory,
    )

"""
Factory for creating database instances.

This module provides factory functions for creating database instances,
including graph databases, vector stores, and composite databases.
"""

import os
from typing import Optional, Type, Union

from dotenv import load_dotenv

from alpha_recall.db.base import GraphDatabase
from alpha_recall.db.neo4j_db import Neo4jDatabase
from alpha_recall.db.memgraph_db import MemgraphDatabase
from alpha_recall.db.semantic_search import SemanticSearch
from alpha_recall.db.vector_store import VectorStore
from alpha_recall.db.composite_db import CompositeDatabase
from alpha_recall.logging_utils import get_logger

# Load environment variables
load_dotenv()

# Get logger
logger = get_logger("db_factory")

# Database configuration from environment
GRAPH_DB_TYPE = os.environ.get("GRAPH_DB", "neo4j").lower()
VECTOR_STORE_URL = os.environ.get("VECTOR_STORE_URL", "http://localhost:6333")
VECTOR_STORE_COLLECTION = os.environ.get("VECTOR_STORE_COLLECTION", "alpha_recall_observations")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


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
    logger.info(f"Creating vector store with URL: {VECTOR_STORE_URL}")
    return VectorStore(
        qdrant_url=VECTOR_STORE_URL,
        collection_name=VECTOR_STORE_COLLECTION,
        model_name=EMBEDDING_MODEL
    )


async def create_db_instance() -> Union[GraphDatabase, CompositeDatabase]:
    """
    Create a composite database instance with both graph database and vector store.
    
    Returns:
        CompositeDatabase instance that combines graph database and vector store
        
    Raises:
        ValueError: If the configured database type is not supported
    """
    # Create graph database
    graph_db = await create_graph_db()
    
    # Create vector store
    vector_store = create_vector_store()
    
    # Create composite database
    logger.info("Creating composite database instance")
    return CompositeDatabase(graph_db=graph_db, semantic_search=vector_store)

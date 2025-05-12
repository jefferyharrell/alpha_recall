"""
Factory for creating graph database instances.
"""

import os
from typing import Optional, Type

from dotenv import load_dotenv

from alpha_recall.db.base import GraphDatabase
from alpha_recall.db.neo4j_db import Neo4jDatabase
from alpha_recall.logging_utils import get_logger

# Load environment variables
load_dotenv()

# Get logger
logger = get_logger("db_factory")

# Database type from environment
GRAPH_DB_TYPE = os.environ.get("GRAPH_DB", "neo4j").lower()


async def create_db_instance() -> GraphDatabase:
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
    # Future database types can be added here
    # elif GRAPH_DB_TYPE == "memgraph":
    #     logger.info("Creating Memgraph database instance")
    #     db = MemgraphDatabase()
    else:
        logger.error(f"Unsupported database type: {GRAPH_DB_TYPE}")
        raise ValueError(f"Unsupported database type: {GRAPH_DB_TYPE}")
    
    # Connect to the database
    await db.connect()
    return db

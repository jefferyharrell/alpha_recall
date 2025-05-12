#!/usr/bin/env python
"""
Migration script to populate the vector store with existing observations from Neo4j.

This script:
1. Fetches all observations from the Neo4j database
2. Generates embeddings for each observation
3. Stores them in the Qdrant vector store with proper IDs and metadata

Usage:
    python -m alpha_recall.migrate_observations
"""

import asyncio
import os
from typing import Dict, List, Any

from dotenv import load_dotenv

from alpha_recall.db.factory import create_db_instance, create_vector_store
from alpha_recall.db.neo4j_db import Neo4jDatabase
from alpha_recall.db.vector_store import VectorStore
from alpha_recall.logging_utils import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


async def fetch_all_observations(db: Neo4jDatabase) -> List[Dict[str, Any]]:
    """
    Fetch all observations from Neo4j.
    
    Returns:
        List of dictionaries containing observation data
    """
    logger.info("Fetching all observations from Neo4j...")
    
    query = """
    MATCH (e:Entity)-[:HAS_OBSERVATION]->(o:Observation)
    RETURN e.name AS entity_name, ID(e) AS entity_id, 
           ID(o) AS observation_id, o.content AS content,
           o.created_at AS created_at
    """
    
    result = await db.execute_query(query)
    logger.info(f"Found {len(result)} observations")
    return result


async def migrate_observations():
    """
    Main migration function to move observations from Neo4j to Qdrant.
    """
    logger.info("Starting observation migration to vector store")
    
    # Create Neo4j database instance directly (not through composite)
    neo4j_uri = os.getenv("GRAPH_DB_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("GRAPH_DB_USER", "neo4j")
    neo4j_password = os.getenv("GRAPH_DB_PASSWORD", "password")
    
    neo4j_db = Neo4jDatabase(neo4j_uri, neo4j_user, neo4j_password)
    await neo4j_db.connect()
    
    # Create vector store instance
    vector_store = create_vector_store()
    
    try:
        # Fetch all observations
        observations = await fetch_all_observations(neo4j_db)
        
        logger.info(f"Migrating {len(observations)} observations to vector store")
        
        # Process observations in batches to avoid memory issues
        batch_size = 100
        total_batches = (len(observations) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(observations))
            batch = observations[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} observations)")
            
            # Process each observation in the batch
            for obs in batch:
                observation_id = obs["observation_id"]
                entity_id = obs["entity_id"]
                content = obs["content"]
                entity_name = obs["entity_name"]
                created_at = obs["created_at"]
                
                # Store in vector store
                success = await vector_store.store_observation(
                    observation_id=observation_id,
                    text=content,
                    entity_id=entity_id,
                    metadata={
                        "entity_name": entity_name,
                        "created_at": created_at
                    }
                )
                
                if not success:
                    logger.warning(f"Failed to migrate observation {observation_id} for entity {entity_name}")
            
            logger.info(f"Completed batch {batch_idx + 1}/{total_batches}")
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        raise
    finally:
        # Close connections
        await neo4j_db.close()


if __name__ == "__main__":
    asyncio.run(migrate_observations())

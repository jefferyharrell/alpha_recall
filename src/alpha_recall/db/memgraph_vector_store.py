"""
Memgraph-based vector store for alpha-recall.

This module provides functionality to:
1. Use Memgraph's native vector search on Observation nodes
2. Generate embeddings from observation text using sentence-transformers
3. Perform semantic search using graph+vector queries
"""

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from alpha_recall.db.semantic_search import SemanticSearch
from alpha_recall.logging_utils import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Default embedding model
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# Default vector dimension for the model
DEFAULT_VECTOR_SIZE = 768  # Dimension of all-mpnet-base-v2 embeddings
# Default vector index name
DEFAULT_INDEX_NAME = "semantic_vector_index"


class MemgraphVectorStore(SemanticSearch):
    """Vector store for observation embeddings using Memgraph's native vector search.

    Implements the SemanticSearch interface using Memgraph as the backend for both
    graph structure and vector search capabilities.
    """

    def __init__(
        self,
        memgraph_uri: str = "bolt://localhost:7687",
        index_name: str = DEFAULT_INDEX_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        """
        Initialize the Memgraph vector store.

        Args:
            memgraph_uri: URI of the Memgraph server
            index_name: Name of the vector index to use
            model_name: Name of the sentence-transformers model to use
        """
        self.memgraph_uri = memgraph_uri
        self.index_name = index_name
        self.model_name = model_name

        # Initialize sentence-transformers model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)

        # Initialize Memgraph driver
        self.driver = GraphDatabase.driver(memgraph_uri)

        # Set vector size
        self.vector_size = DEFAULT_VECTOR_SIZE

        logger.info(f"Memgraph vector store initialized with model: {self.model_name}")

        # Ensure vector index exists
        self._ensure_vector_index_exists()

    def _ensure_vector_index_exists(self) -> None:
        """Ensure that the vector index exists, creating it if necessary."""
        with self.driver.session() as session:
            # Check if index exists
            result = session.run("SHOW INDEX INFO")
            existing_indexes = [record["property"] for record in result if record.get("property") == "semantic_vector"]
            
            if not existing_indexes:
                logger.info(f"Creating vector index: {self.index_name}")
                session.run(f"""
                CREATE VECTOR INDEX {self.index_name} ON :Observation(semantic_vector) 
                WITH CONFIG {{"dimension": {self.vector_size}, "capacity": 1000, "metric": "cos"}}
                """)
                logger.info("Vector index created successfully")
            else:
                logger.info("Vector index already exists")

    async def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the given text using sentence-transformers.

        Args:
            text: Text to embed, can be a single string or a list of strings

        Returns:
            Embeddings as a numpy array
        """
        try:
            # Use sentence-transformers to generate embeddings
            embeddings = self.embedding_model.encode(text)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def store_observation(
        self,
        observation_id: str,
        text: str,
        entity_id: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Store an observation with its embedding by updating the Observation node.

        Args:
            observation_id: Unique ID of the observation (from Memgraph)
            text: Text of the observation
            entity_id: ID of the entity this observation belongs to
            metadata: Additional metadata to store (not used in this implementation)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = await self.embed_text(text)

            with self.driver.session() as session:
                # Find the observation by content and update it with the vector
                result = session.run("""
                MATCH (o:Observation)
                WHERE o.content = $text
                SET o.semantic_vector = $vector
                RETURN count(o) as updated_count
                """, {
                    "text": text,
                    "vector": embedding.tolist()
                })
                
                record = result.single()
                updated_count = record["updated_count"] if record else 0
                
                if updated_count > 0:
                    logger.debug(f"Updated {updated_count} observation(s) with semantic vector")
                    return True
                else:
                    logger.warning(f"No observation found with text: {text[:50]}...")
                    return False
                    
        except Exception as e:
            logger.error(f"Error storing observation vector: {e}")
            return False

    async def search_observations(
        self, query: str, limit: int = 10, entity_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for observations semantically similar to the query using Memgraph vector search.

        Args:
            query: Query text
            limit: Maximum number of results to return
            entity_id: Optional entity ID to filter results (not implemented yet)

        Returns:
            List of matching observations with scores
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embed_text(query)

            with self.driver.session() as session:
                # Perform vector search
                if entity_id:
                    # Search with entity filter
                    result = session.run("""
                    CALL vector_search.search($index_name, $limit, $query_vector) 
                    YIELD node, similarity 
                    MATCH (e:Entity)-[:HAS_OBSERVATION]->(node)
                    WHERE e.name = $entity_name
                    RETURN similarity as score, 
                           node.content as text,
                           e.name as entity_name
                    ORDER BY similarity DESC
                    """, {
                        "index_name": self.index_name,
                        "query_vector": query_embedding.tolist(),
                        "limit": limit,
                        "entity_name": entity_id
                    })
                else:
                    # Search without entity filter
                    result = session.run("""
                    CALL vector_search.search($index_name, $limit, $query_vector) 
                    YIELD node, similarity 
                    MATCH (e:Entity)-[:HAS_OBSERVATION]->(node)
                    RETURN similarity as score, 
                           node.content as text,
                           e.name as entity_name
                    ORDER BY similarity DESC
                    """, {
                        "index_name": self.index_name,
                        "query_vector": query_embedding.tolist(),
                        "limit": limit
                    })

                # Format results
                formatted_results = []
                for record in result:
                    formatted_results.append({
                        "entity_name": record["entity_name"],
                        "text": record["text"],
                        "score": record["score"]
                    })

                return formatted_results
                
        except Exception as e:
            logger.error(f"Error searching observations: {e}")
            return []

    async def delete_observation(self, observation_id: str) -> bool:
        """
        Delete an observation's vector from the store (removes semantic_vector property).

        Args:
            observation_id: ID of the observation to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (o:Observation)
                WHERE id(o) = $observation_id
                REMOVE o.semantic_vector
                RETURN count(o) as updated_count
                """, {"observation_id": int(observation_id)})
                
                record = result.single()
                updated_count = record["updated_count"] if record else 0
                return updated_count > 0
                
        except Exception as e:
            logger.error(f"Error deleting observation vector: {e}")
            return False

    async def delete_entity_observations(self, entity_id: str) -> bool:
        """
        Delete all observation vectors for a given entity.

        Args:
            entity_id: ID of the entity (entity name)

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (e:Entity {name: $entity_name})-[:HAS_OBSERVATION]->(o:Observation)
                REMOVE o.semantic_vector
                RETURN count(o) as updated_count
                """, {"entity_name": entity_id})
                
                record = result.single()
                updated_count = record["updated_count"] if record else 0
                logger.info(f"Removed vectors from {updated_count} observations for entity: {entity_id}")
                return updated_count > 0
                
        except Exception as e:
            logger.error(f"Error deleting entity observation vectors: {e}")
            return False

    def close(self):
        """Close the Memgraph driver connection."""
        if self.driver:
            self.driver.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
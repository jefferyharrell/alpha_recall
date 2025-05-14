"""
Vector store module for alpha-recall using Qdrant and sentence-transformers.

This module provides functionality to:
1. Create and manage a Qdrant collection for storing observation embeddings
2. Generate embeddings from observation text using sentence-transformers
3. Perform semantic search on observations
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from alpha_recall.db.semantic_search import SemanticSearch
from alpha_recall.logging_utils import get_logger

logger = get_logger(__name__)

# Default embedding model
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
# Default vector dimension for the model
DEFAULT_VECTOR_SIZE = 384  # Dimension of all-MiniLM-L6-v2 embeddings
# Default collection name
DEFAULT_COLLECTION_NAME = "alpha_recall_observations"


class VectorStore(SemanticSearch):
    """Vector store for observation embeddings using Qdrant and sentence-transformers.
    
    Implements the SemanticSearch interface using Qdrant as the backend.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        """
        Initialize the vector store.

        Args:
            qdrant_url: URL of the Qdrant server
            collection_name: Name of the collection to use
            model_name: Name of the sentence-transformers model to use
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=qdrant_url)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.vector_size = DEFAULT_VECTOR_SIZE
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """Ensure that the collection exists, creating it if necessary."""
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
    
    async def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the given text.

        Args:
            text: Text to embed, can be a single string or a list of strings

        Returns:
            Embeddings as a numpy array
        """
        # Note: sentence-transformers doesn't have async methods, so we're just wrapping it
        return self.model.encode(text)
    
    async def store_observation(
        self, 
        observation_id: str, 
        text: str, 
        entity_id: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store an observation with its embedding.

        Args:
            observation_id: Unique ID of the observation (from Neo4j)
            text: Text of the observation
            entity_id: ID of the entity this observation belongs to
            metadata: Additional metadata to store

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = await self.embed_text(text)
            
            # Prepare payload
            payload = {
                "observation_id": observation_id,
                "entity_id": entity_id,
                "text": text,
            }
            
            # Add metadata if provided
            if metadata:
                payload.update(metadata)
            
            # Convert string ID to integer for Qdrant (which expects unsigned int or UUID)
            try:
                # Try to convert to integer first
                point_id = int(observation_id)
            except ValueError:
                # If it's not a valid integer, use it as is (might be UUID)
                point_id = observation_id
                
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,  # Use the converted ID
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                ],
            )
            return True
        except Exception as e:
            logger.error(f"Error storing observation: {e}")
            return False
    
    async def search_observations(
        self, 
        query: str, 
        limit: int = 10, 
        entity_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for observations semantically similar to the query.

        Args:
            query: Query text
            limit: Maximum number of results to return
            entity_id: Optional entity ID to filter results

        Returns:
            List of matching observations with scores
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embed_text(query)
            
            # Prepare filter if entity_id is provided
            search_filter = None
            if entity_id:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="entity_id",
                            match=models.MatchValue(value=entity_id),
                        )
                    ]
                )
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=search_filter,
            )
            
            # Format results
            formatted_results = []
            for res in results:
                formatted_results.append({
                    "text": res.payload.get("text"),
                    "score": res.score,
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching observations: {e}")
            return []
    
    async def delete_observation(self, observation_id: str) -> bool:
        """
        Delete an observation from the vector store.

        Args:
            observation_id: ID of the observation to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[observation_id],
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting observation: {e}")
            return False
    
    async def delete_entity_observations(self, entity_id: str) -> bool:
        """
        Delete all observations for a given entity.

        Args:
            entity_id: ID of the entity

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="entity_id",
                                match=models.MatchValue(value=entity_id),
                            )
                        ]
                    )
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting entity observations: {e}")
            return False

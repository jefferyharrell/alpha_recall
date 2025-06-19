"""
Composite database that combines graph database, semantic search, and short-term memory capabilities.
"""

import os
import platform
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from alpha_recall.db.base import GraphDatabase
from alpha_recall.db.redis_db import RedisShortTermMemory
from alpha_recall.db.semantic_search import SemanticSearch
from alpha_recall.logging_utils import get_logger

logger = get_logger(__name__)


class CompositeDatabase:
    async def recency_search(self, limit: int = 10) -> list:
        """
        Return the N most recent observations within the given time span.
        Args:
            span: A string representing the time span (e.g., '1h', '1d')
            limit: Maximum number of results to return (default 10)
        Returns:
            List of recent observations
        """
        if hasattr(self.graph_db, "recency_search"):
            return await self.graph_db.recency_search(limit)
        else:
            logger.error("recency_search not implemented in graph_db backend")
            return []

    """
    Composite database that combines graph database and semantic search capabilities.
    
    This class delegates graph operations to a GraphDatabase implementation
    and semantic search operations to a SemanticSearch implementation.
    """

    def __init__(
        self,
        graph_db: GraphDatabase,
        semantic_search: SemanticSearch,
        shortterm_memory: Optional[RedisShortTermMemory] = None,
    ):
        """
        Initialize the composite database.

        Args:
            graph_db: Implementation of GraphDatabase
            semantic_search: Implementation of SemanticSearch
            shortterm_memory: Optional Redis-based short-term memory implementation
        """
        self.graph_db = graph_db
        self.search_engine = semantic_search
        self.shortterm_memory = shortterm_memory

    async def create_entity(
        self, name: str, entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new entity in the graph database.

        Args:
            name: Name of the entity
            entity_type: Optional type of the entity

        Returns:
            Dictionary representing the created entity
        """
        return await self.graph_db.create_entity(name, entity_type)

    async def add_observation(
        self, entity_name: str, observation: str
    ) -> Dict[str, Any]:
        """
        Add an observation to an entity in both the graph database and vector store.

        Args:
            entity_name: Name of the entity
            observation: Content of the observation

        Returns:
            Dictionary representing the updated entity with the new observation
        """
        # First add to graph database
        result = await self.graph_db.add_observation(entity_name, observation)

        # Then add to vector store
        observation_id = result["observation"]["id"]
        entity_id = result["entity"]["id"]

        # Store in vector database for semantic search
        await self.search_engine.store_observation(
            observation_id=observation_id,
            text=observation,
            entity_id=entity_id,
            metadata={
                "entity_name": entity_name,
                "created_at": result["observation"]["created_at"],
            },
        )

        return result

    async def create_relationship(
        self, source_entity: str, target_entity: str, relationship_type: str
    ) -> Dict[str, Any]:
        """
        Create a relationship between two entities in the graph database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relationship_type: Type of the relationship

        Returns:
            Dictionary representing the created relationship
        """
        return await self.graph_db.create_relationship(
            source_entity, target_entity, relationship_type
        )

    async def get_entity(
        self, entity_name: str, depth: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entity and its relationships from the graph database.

        Args:
            entity_name: Name of the entity to retrieve
            depth: How many relationship hops to include

        Returns:
            Dictionary representing the entity and its relationships,
            or None if the entity doesn't exist
        """
        return await self.graph_db.get_entity(entity_name, depth)

    async def delete_entity(self, name: str) -> Dict[str, Any]:
        """
        Delete an entity and all its relationships from both databases.

        Args:
            name: Name of the entity to delete

        Returns:
            Dictionary containing the deletion status and details
        """
        # First get the entity to get its ID
        entity = await self.graph_db.get_entity(name, depth=0)
        if not entity:
            return {"success": False, "error": f"Entity '{name}' not found"}

        entity_id = entity["id"]

        # Delete from vector store
        vector_result = await self.search_engine.delete_entity_observations(entity_id)

        # Delete from graph database
        graph_result = await self.graph_db.delete_entity(name)

        # Combine results
        graph_result["vector_store_success"] = vector_result

        return graph_result

    async def semantic_search(
        self, query: str, limit: int = 10, entity_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform semantic search on observations.

        Args:
            query: Query text
            limit: Maximum number of results to return
            entity_name: Optional entity name to filter results

        Returns:
            List of matching observations with scores
        """
        entity_id = None
        if entity_name:
            # Get entity ID if entity name is provided
            entity = await self.graph_db.get_entity(entity_name, depth=0)
            if entity:
                entity_id = entity["id"]

        return await self.search_engine.search_observations(
            query=query, limit=limit, entity_id=entity_id
        )

    async def remember_shortterm(
        self, content: str, client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a short-term memory with automatic TTL expiration and return relevant memories.

        Args:
            content: The memory content to store
            client_info: Optional information about the client/source

        Returns:
            Dictionary containing information about the stored memory and relevant memories
        """
        if not self.shortterm_memory:
            logger.warning("Short-term memory storage not configured")
            return {
                "success": False,
                "error": "Short-term memory storage not configured",
            }

        try:
            # If client_info is not provided, try to detect it
            if not client_info:
                client_info = self._detect_client()

            # Store the memory
            result = await self.shortterm_memory.store_memory(content, client_info)
            
            # If storage was successful, retrieve relevant memories
            if result.get("success", True):  # Default to True for backward compatibility
                # Get relevant memories using a combination of semantic search and recency
                relevant_memories = await self._get_relevant_memories(content, limit=5)
                result["relevant_memories"] = relevant_memories
                
            return result
        except Exception as e:
            logger.error(f"Failed to store short-term memory: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_shortterm_memories(
        self, through_the_last: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent short-term memories.

        Args:
            through_the_last: Optional time window (e.g., '2h', '1d')
            limit: Maximum number of memories to return

        Returns:
            List of recent memories, newest first
        """
        if not self.shortterm_memory:
            logger.warning("Short-term memory storage not configured")
            return []

        try:
            # Retrieve memories
            return await self.shortterm_memory.get_recent_memories(
                through_the_last=through_the_last, limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to retrieve short-term memories: {str(e)}")
            return []
    
    async def semantic_search_shortterm(
        self, 
        query: str, 
        limit: int = 10,
        through_the_last: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on short-term memories.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            through_the_last: Optional time window filter (e.g., '2h', '1d')
            
        Returns:
            List of semantically similar memories with scores
        """
        if not self.shortterm_memory:
            logger.warning("Short-term memory storage not configured")
            return []
            
        # Check if the short-term memory has semantic search capability
        if hasattr(self.shortterm_memory, 'semantic_search_memories'):
            try:
                return await self.shortterm_memory.semantic_search_memories(
                    query=query,
                    limit=limit,
                    through_the_last=through_the_last
                )
            except Exception as e:
                logger.error(f"Failed to perform semantic search on short-term memories: {str(e)}")
                # Fall back to regular retrieval
                return await self.get_shortterm_memories(through_the_last, limit)
        else:
            # Fall back to regular retrieval if semantic search not available
            logger.info("Semantic search not available for short-term memories, using time-based retrieval")
            return await self.get_shortterm_memories(through_the_last, limit)

    async def emotional_search_shortterm(
        self, 
        query: str, 
        limit: int = 10,
        through_the_last: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform emotional search on short-term memories.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            through_the_last: Optional time window filter (e.g., '2h', '1d')
            
        Returns:
            List of emotionally similar memories with scores
        """
        if not self.shortterm_memory:
            logger.warning("Short-term memory storage not configured")
            return []
            
        # Check if the short-term memory has emotional search capability
        if hasattr(self.shortterm_memory, 'emotional_search_memories'):
            try:
                return await self.shortterm_memory.emotional_search_memories(
                    query=query,
                    limit=limit,
                    through_the_last=through_the_last
                )
            except Exception as e:
                logger.error(f"Failed to perform emotional search on short-term memories: {str(e)}")
                # Fall back to regular retrieval
                return await self.get_shortterm_memories(through_the_last, limit)
        else:
            # Fall back to regular retrieval if emotional search not available
            logger.info("Emotional search not available for short-term memories, using time-based retrieval")
            return await self.get_shortterm_memories(through_the_last, limit)

    def _detect_client(self) -> Dict[str, str]:
        """
        Detect information about the current client environment.

        Returns:
            Dict with client information
        """
        client_info = {}

        # Try to get environment-specific information
        client_name = os.environ.get("CLIENT_NAME", "unknown")
        client_info["client_name"] = client_name

        return client_info
    
    async def _get_relevant_memories(self, query: str, limit: int = 5, include_emotional: bool = True) -> List[Dict[str, Any]]:
        """
        Get the most relevant memories based on a combination of semantic similarity, emotional similarity, and recency.
        
        This method implements a relevance algorithm that combines:
        - Semantic similarity score (from vector search)
        - Emotional similarity score (from emotional vector search) 
        - Recency (how recent the memory is)
        
        The formula is: relevance_score = 0.5 * semantic_similarity + 0.2 * emotional_similarity + 0.3 * recency_score
        
        Args:
            query: The query text to find relevant memories for
            limit: Maximum number of relevant memories to return
            include_emotional: Whether to include emotional similarity in scoring
            
        Returns:
            List of the most relevant memories with combined scores
        """
        # Get semantically similar memories (up to 3x the limit to have enough for reranking)
        semantic_results = await self.semantic_search_shortterm(
            query=query, 
            limit=limit * 3
        )
        logger.info(f"Semantic search returned {len(semantic_results)} results")
        if semantic_results:
            logger.debug(f"First semantic result ID: {semantic_results[0].get('id')}, score: {semantic_results[0].get('similarity_score')}")
        
        # Get emotionally similar memories if enabled
        emotional_results = []
        if include_emotional:
            try:
                emotional_results = await self.emotional_search_shortterm(
                    query=query,
                    limit=limit * 3
                )
                logger.info(f"Emotional search returned {len(emotional_results)} results")
                if emotional_results:
                    logger.debug(f"First emotional result ID: {emotional_results[0].get('id')}, score: {emotional_results[0].get('emotional_score')}")
            except Exception as e:
                logger.warning(f"Emotional search failed: {str(e)}")
                emotional_results = []
        
        # Get recent memories for recency scoring
        recent_memories = await self.get_shortterm_memories(limit=limit * 3)
        logger.info(f"Recent memories returned {len(recent_memories)} results")
        
        # Create a map of memory IDs to memories for quick lookup
        memory_map = {}
        for memory in semantic_results:
            memory_id = memory.get("id")
            if memory_id:
                memory_map[memory_id] = memory
                # Initialize emotional_score to default if not present
                memory.setdefault("emotional_score", 1.0)
        
        # Add emotional results and merge emotional scores
        for memory in emotional_results:
            memory_id = memory.get("id")
            emotional_score = memory.get("emotional_score", 1.0)
            if memory_id:
                if memory_id in memory_map:
                    # Merge emotional score into existing memory
                    memory_map[memory_id]["emotional_score"] = emotional_score
                    logger.debug(f"Merged emotional score {emotional_score} for memory {memory_id}")
                else:
                    # Add new memory from emotional results
                    memory_map[memory_id] = memory
                    # Set a low default similarity score for memories not in semantic results
                    memory["similarity_score"] = 1.0
                    logger.debug(f"Added new memory from emotional results: {memory_id}")
                
        # Add any recent memories not in either semantic or emotional results
        for memory in recent_memories:
            memory_id = memory.get("id")
            if memory_id and memory_id not in memory_map:
                memory_map[memory_id] = memory
                # Set default scores for memories not in search results
                memory["similarity_score"] = 1.0
                memory["emotional_score"] = 1.0
            elif memory_id and memory_id in memory_map:
                # Ensure emotional_score is set for memories already in map
                memory_map[memory_id].setdefault("emotional_score", 1.0)
        
        # Calculate relevance scores
        current_time = datetime.utcnow()
        scored_memories = []
        
        for memory_id, memory in memory_map.items():
            # Get semantic similarity score (0-1 range, where lower is better in Redis)
            # Convert to 0-1 range where higher is better
            similarity_score = memory.get("similarity_score", 1.0)
            # Redis returns cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity: 1 - (distance / 2)
            semantic_score = max(0, 1 - (similarity_score / 2))
            
            # Get emotional similarity score and convert similarly
            emotional_score_raw = memory.get("emotional_score", 1.0)
            emotional_score = max(0, 1 - (emotional_score_raw / 2)) if include_emotional else 0.5
            
            # Calculate recency score (0-1 range, where 1 is most recent)
            created_at_str = memory.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    # Calculate age in hours
                    age_hours = (current_time - created_at).total_seconds() / 3600
                    # Use exponential decay: e^(-age/24) gives ~0.37 after 24 hours
                    recency_score = np.exp(-age_hours / 24)
                except Exception as e:
                    logger.warning(f"Failed to parse created_at for memory {memory_id}: {e}")
                    recency_score = 0.0
            else:
                recency_score = 0.0
            
            # Calculate combined relevance score
            # Weights: 50% semantic similarity, 20% emotional similarity, 30% recency
            if include_emotional:
                relevance_score = 0.5 * semantic_score + 0.2 * emotional_score + 0.3 * recency_score
                if memory_id == next(iter(memory_map.keys())):  # Log for first memory only
                    logger.info(f"Using emotional scoring: 0.5*{semantic_score} + 0.2*{emotional_score} + 0.3*{recency_score} = {relevance_score}")
            else:
                # If emotional scoring is disabled, use original weights
                relevance_score = 0.7 * semantic_score + 0.3 * recency_score
                if memory_id == next(iter(memory_map.keys())):  # Log for first memory only
                    logger.info(f"Using legacy scoring: 0.7*{semantic_score} + 0.3*{recency_score} = {relevance_score}")
            
            memory["relevance_score"] = relevance_score
            memory["semantic_score"] = semantic_score
            memory["emotional_score"] = emotional_score
            memory["recency_score"] = recency_score
            scored_memories.append(memory)
        
        # Sort by relevance score (highest first)
        scored_memories.sort(key=lambda m: m.get("relevance_score", 0), reverse=True)
        
        # Return only the top memories
        return scored_memories[:limit]

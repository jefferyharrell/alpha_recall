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
from alpha_recall.db.narrative_memory import NarrativeMemory
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
        narrative_memory: Optional[NarrativeMemory] = None,
    ):
        """
        Initialize the composite database.

        Args:
            graph_db: Implementation of GraphDatabase
            semantic_search: Implementation of SemanticSearch
            shortterm_memory: Optional Redis-based short-term memory implementation
            narrative_memory: Optional narrative memory implementation
        """
        self.graph_db = graph_db
        self.search_engine = semantic_search
        self.shortterm_memory = shortterm_memory
        self.narrative_memory = narrative_memory

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
                # Pass the new memory's ID to exclude it from results
                new_memory_id = result.get("id")
                relevant_memories = await self._get_relevant_memories(content, limit=5, exclude_id=new_memory_id)
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
    
    async def _get_relevant_memories(self, query: str, limit: int = 5, include_emotional: bool = True, exclude_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get relevant memories using both semantic and emotional search.
        
        This method returns the top memories from:
        - Semantic similarity search (up to 'limit' results)
        - Emotional similarity search (up to 'limit' results) if enabled
        
        The results are returned as two separate groups rather than trying to merge scores,
        providing transparency about which type of search surfaced each memory.
        
        Args:
            query: The query text to find relevant memories for
            limit: Maximum number of memories to return from each search type
            include_emotional: Whether to include emotional similarity search
            exclude_id: Optional memory ID to exclude from results (e.g., the memory being created)
            
        Returns:
            List of relevant memories with their respective similarity scores
        """
        all_results = []
        seen_ids = set()
        
        # If we have an ID to exclude, add it to seen_ids
        if exclude_id:
            seen_ids.add(exclude_id)
            logger.debug(f"Excluding memory ID from results: {exclude_id}")
        
        # Get semantically similar memories
        semantic_results = await self.semantic_search_shortterm(
            query=query, 
            limit=limit * 2  # Get more to account for potential duplicates
        )
        logger.info(f"Semantic search returned {len(semantic_results)} results")
        
        # Process semantic results
        for memory in semantic_results:
            memory_id = memory.get("id")
            if memory_id and memory_id not in seen_ids:
                # Convert Redis cosine distance to similarity score
                distance = memory.get("similarity_score", 1.0)
                similarity = max(0, 1 - (distance / 2))
                
                # Add search type and score
                memory["search_type"] = "semantic"
                memory["score"] = round(similarity, 3)
                all_results.append(memory)
                seen_ids.add(memory_id)
                
                # Debug log the first result
                if len(all_results) == 1:
                    logger.debug(f"First semantic result - ID: {memory_id}, raw distance: {distance}, similarity: {similarity}")
        
        # Get emotionally similar memories if enabled
        if include_emotional:
            try:
                emotional_results = await self.emotional_search_shortterm(
                    query=query,
                    limit=limit * 2  # Get more to account for potential duplicates
                )
                logger.info(f"Emotional search returned {len(emotional_results)} results")
                
                # Process emotional results
                for memory in emotional_results:
                    memory_id = memory.get("id")
                    if memory_id and memory_id not in seen_ids:
                        # Convert Redis cosine distance to similarity score
                        distance = memory.get("emotional_score", 1.0)
                        similarity = max(0, 1 - (distance / 2))
                        
                        # Add search type and score
                        memory["search_type"] = "emotional"
                        memory["score"] = round(similarity, 3)
                        all_results.append(memory)
                        seen_ids.add(memory_id)
                        
                        # Debug log the first emotional result
                        if memory.get("search_type") == "emotional" and len([m for m in all_results if m.get("search_type") == "emotional"]) == 1:
                            logger.debug(f"First emotional result - ID: {memory_id}, raw distance: {distance}, similarity: {similarity}")
                    
            except Exception as e:
                logger.warning(f"Emotional search failed: {str(e)}")
        
        # Sort all results by score (highest first)
        all_results.sort(key=lambda m: m.get("score", 0), reverse=True)
        
        # Return only the top N results
        return all_results[:limit]

    async def store_narrative(
        self,
        title: str,
        paragraphs: List[str],
        participants: List[str],
        tags: Optional[List[str]] = None,
        outcome: str = "ongoing",
        references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Store a narrative story with hybrid storage (Redis content + Memgraph relationships).

        Args:
            title: Story title
            paragraphs: List of paragraph texts
            participants: List of participant names
            tags: Optional list of tags/topics
            outcome: Story outcome ("breakthrough", "resolution", "ongoing")
            references: Optional list of story_ids this story references

        Returns:
            Dictionary with story information and storage status
        """
        if not self.narrative_memory:
            logger.warning("Narrative memory storage not configured")
            return {
                "success": False,
                "error": "Narrative memory storage not configured",
            }

        try:
            return await self.narrative_memory.store_story(
                title=title,
                paragraphs=paragraphs,
                participants=participants,
                tags=tags,
                outcome=outcome,
                references=references,
            )
        except Exception as e:
            logger.error(f"Failed to store narrative: {str(e)}")
            return {"success": False, "error": str(e)}

    async def search_narratives(
        self,
        query: str,
        search_type: str = "semantic",  # "semantic", "emotional", "both"
        granularity: str = "story",     # "story", "paragraph", "both"  
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search narrative stories using vector similarity.

        Args:
            query: Search query text
            search_type: Type of search ("semantic", "emotional", "both")
            granularity: Search granularity ("story", "paragraph", "both")
            limit: Maximum number of results

        Returns:
            List of matching stories/paragraphs with similarity scores
        """
        if not self.narrative_memory:
            logger.warning("Narrative memory storage not configured")
            return []

        try:
            return await self.narrative_memory.search_stories(
                query=query,
                search_type=search_type,
                granularity=granularity,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Failed to search narratives: {str(e)}")
            return []

    async def get_narrative(self, story_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete narrative story by ID.

        Args:
            story_id: Unique story identifier

        Returns:
            Complete story data or None if not found
        """
        if not self.narrative_memory:
            logger.warning("Narrative memory storage not configured")
            return None

        try:
            return await self.narrative_memory.get_story(story_id)
        except Exception as e:
            logger.error(f"Failed to retrieve narrative {story_id}: {str(e)}")
            return None

    async def list_narratives(
        self,
        limit: int = 10,
        offset: int = 0,
        since: Optional[str] = None,
        participants: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List narrative stories chronologically with optional filtering.

        Args:
            limit: Maximum number of stories to return
            offset: Number of stories to skip (for pagination)  
            since: Time window filter (e.g., "2d", "1w", "1m")
            participants: Filter by participants (AND logic)
            tags: Filter by tags (AND logic)
            outcome: Filter by outcome type

        Returns:
            Dictionary with stories list and pagination info
        """
        if not self.narrative_memory:
            return {
                "stories": [],
                "total_count": 0,
                "returned_count": 0,
                "error": "Narrative memory not available"
            }

        try:
            return await self.narrative_memory.list_stories(
                limit=limit,
                offset=offset,
                since=since,
                participants=participants,
                tags=tags,
                outcome=outcome
            )
        except Exception as e:
            logger.error(f"Failed to list narratives: {str(e)}")
            return {
                "stories": [],
                "total_count": 0,
                "returned_count": 0,
                "error": str(e)
            }

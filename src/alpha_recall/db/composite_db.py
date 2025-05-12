"""
Composite database that combines graph database and semantic search capabilities.
"""

from typing import Any, Dict, List, Optional, Union

from alpha_recall.db.base import GraphDatabase
from alpha_recall.db.semantic_search import SemanticSearch
from alpha_recall.logging_utils import get_logger

logger = get_logger(__name__)


class CompositeDatabase:
    """
    Composite database that combines graph database and semantic search capabilities.
    
    This class delegates graph operations to a GraphDatabase implementation
    and semantic search operations to a SemanticSearch implementation.
    """
    
    def __init__(
        self,
        graph_db: GraphDatabase,
        semantic_search: SemanticSearch
    ):
        """
        Initialize the composite database.
        
        Args:
            graph_db: Implementation of GraphDatabase
            semantic_search: Implementation of SemanticSearch
        """
        self.graph_db = graph_db
        self.semantic_search = semantic_search
    
    async def create_entity(
        self, 
        name: str, 
        entity_type: Optional[str] = None
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
        self, 
        entity_name: str, 
        observation: str
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
        await self.semantic_search.store_observation(
            observation_id=observation_id,
            text=observation,
            entity_id=entity_id,
            metadata={
                "entity_name": entity_name,
                "created_at": result["observation"]["created_at"]
            }
        )
        
        return result
    
    async def create_relationship(
        self, 
        source_entity: str, 
        target_entity: str, 
        relationship_type: str
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
        self, 
        entity_name: str, 
        depth: int = 1
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
        vector_result = await self.semantic_search.delete_entity_observations(entity_id)
        
        # Delete from graph database
        graph_result = await self.graph_db.delete_entity(name)
        
        # Combine results
        graph_result["vector_store_success"] = vector_result
        
        return graph_result
    
    async def semantic_search(
        self, 
        query: str, 
        limit: int = 10, 
        entity_name: Optional[str] = None
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
        
        return await self.semantic_search.search_observations(
            query=query,
            limit=limit,
            entity_id=entity_id
        )

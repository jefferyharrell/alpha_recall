"""
Abstract base class for graph database operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import timedelta


class GraphDatabase(ABC):
    """
    Abstract base class defining the interface for graph database operations.
    
    This allows for different graph database implementations (Neo4j, Memgraph, etc.)
    while maintaining a consistent interface for the application.
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish a connection to the database.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        Close the database connection.
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if the database connection is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query against the database.
        
        Args:
            query: The query string to execute
            parameters: Optional parameters for the query
            
        Returns:
            List of records as dictionaries
        """
        pass
    
    @abstractmethod
    async def create_entity(
        self, 
        name: str, 
        entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new entity in the graph.
        
        Args:
            name: Name of the entity
            entity_type: Optional type of the entity
            
        Returns:
            Dictionary representing the created entity
        """
        pass
    
    @abstractmethod
    async def add_observation(
        self, 
        entity_name: str, 
        observation: str
    ) -> Dict[str, Any]:
        """
        Add an observation to an entity.
        
        Args:
            entity_name: Name of the entity
            observation: Content of the observation
            
        Returns:
            Dictionary representing the updated entity
        """
        pass
    
    @abstractmethod
    async def create_relationship(
        self, 
        source_entity: str, 
        target_entity: str, 
        relationship_type: str
    ) -> Dict[str, Any]:
        """
        Create a relationship between two entities.
        
        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relationship_type: Type of the relationship
            
        Returns:
            Dictionary representing the created relationship
        """
        pass
    
    @abstractmethod
    async def get_entity(
        self, 
        entity_name: str, 
        depth: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entity and its relationships from the graph.
        
        Args:
            entity_name: Name of the entity to retrieve
            depth: How many relationship hops to include
                0: Only the entity itself
                1: Entity and direct relationships
                2+: Entity and extended network
                
        Returns:
            Dictionary representing the entity and its relationships,
            or None if the entity doesn't exist
        """
        pass

    @abstractmethod
    async def delete_entity(
        self,
        name: str
    ) -> Dict[str, Any]:
        """
        Delete an entity and all its relationships (and attached observations) from the graph.
        
        Args:
            name: Name of the entity to delete
        Returns:
            Dictionary containing the deletion status and details
        """
        pass

    @abstractmethod
    async def recency_search(
        self,
        limit: int = 10
    ) -> list:
        """
        Return the N most recent observations within the given time span.
        Args:
            span: A string representing the time span (e.g., '1h', '1d')
            limit: Maximum number of results to return (default 10)
        Returns:
            List of recent observations
        """
        pass
        
    @abstractmethod
    async def remember_shortterm(
        self,
        content: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a short-term memory with automatic TTL expiration.
        
        Args:
            content: The memory content to store
            client_info: Optional information about the client/source
            
        Returns:
            Dictionary containing information about the stored memory
        """
        pass
        
    @abstractmethod
    async def get_shortterm_memories(
        self,
        through_the_last: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent short-term memories.
        
        Args:
            through_the_last: Optional time window (e.g., '2h', '1d')
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memories, newest first
        """
        pass

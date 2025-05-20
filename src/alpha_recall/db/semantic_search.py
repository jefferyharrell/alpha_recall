"""
Abstract base class for semantic search functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class SemanticSearch(ABC):
    """
    Abstract base class for semantic search functionality.

    This class defines the interface for semantic search operations
    that can be implemented by different vector database backends.
    """

    @abstractmethod
    async def store_observation(
        self,
        observation_id: str,
        text: str,
        entity_id: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Store an observation with its embedding.

        Args:
            observation_id: Unique ID of the observation
            text: Text of the observation
            entity_id: ID of the entity this observation belongs to
            metadata: Additional metadata to store

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def search_observations(
        self, query: str, limit: int = 10, entity_id: Optional[str] = None
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
        pass

    @abstractmethod
    async def delete_observation(self, observation_id: str) -> bool:
        """
        Delete an observation from the vector store.

        Args:
            observation_id: ID of the observation to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_entity_observations(self, entity_id: str) -> bool:
        """
        Delete all observations for a given entity.

        Args:
            entity_id: ID of the entity

        Returns:
            True if successful, False otherwise
        """
        pass

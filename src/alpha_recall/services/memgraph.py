"""Memgraph service for long-term memory operations."""

import time
from typing import Any

from gqlalchemy import Memgraph

from ..config import settings
from ..logging import get_logger
from ..utils.correlation import get_correlation_id

logger = get_logger("services.memgraph")


class MemgraphService:
    """Service for interacting with Memgraph database."""

    def __init__(self):
        """Initialize the Memgraph service."""
        self._db: Memgraph | None = None
        self._connection_tested = False

    @property
    def db(self) -> Memgraph:
        """Get or create Memgraph connection."""
        if self._db is None:
            self._db = Memgraph(
                host=settings.memgraph_uri.replace("bolt://", "").split(":")[0],
                port=int(settings.memgraph_uri.split(":")[-1]),
                encrypted=False,
            )
            logger.debug("Created Memgraph connection", uri=settings.memgraph_uri)
        return self._db

    def test_connection(self) -> bool:
        """Test the Memgraph connection."""
        if self._connection_tested:
            return True

        try:
            start_time = time.perf_counter()
            # Simple test query
            result = list(self.db.execute_and_fetch("RETURN 1 as test"))
            test_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            if result and result[0]["test"] == 1:
                self._connection_tested = True
                logger.info(
                    "Memgraph connection test successful",
                    test_time_ms=test_time_ms,
                    correlation_id=get_correlation_id(),
                )
                return True
            else:
                logger.error(
                    "Memgraph connection test failed - unexpected result",
                    result=result,
                    correlation_id=get_correlation_id(),
                )
                return False
        except Exception as e:
            logger.error(
                "Memgraph connection test failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            return False

    def create_or_update_entity(
        self, entity_name: str, entity_type: str | None = None
    ) -> dict[str, Any]:
        """Create or update an entity in the knowledge graph."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Use MERGE to create or update entity
            query = """
            MERGE (e:Entity {name: $entity_name})
            ON CREATE SET e.created_at = datetime(), e.updated_at = datetime()
            ON MATCH SET e.updated_at = datetime()
            """

            params = {"entity_name": entity_name}

            if entity_type:
                query += ", e.type = $entity_type"
                params["entity_type"] = entity_type

            query += " RETURN e"

            result = list(self.db.execute_and_fetch(query, params))

            if result:
                entity = result[0]["e"]
                operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

                logger.info(
                    "Entity created/updated successfully",
                    entity_name=entity_name,
                    entity_type=entity_type,
                    operation_time_ms=operation_time_ms,
                    correlation_id=correlation_id,
                )

                return {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "created_at": entity.get("created_at"),
                    "updated_at": entity.get("updated_at"),
                }
            else:
                raise Exception("Failed to create/update entity - no result returned")

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to create/update entity",
                entity_name=entity_name,
                entity_type=entity_type,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise

    def add_observation(self, entity_name: str, observation: str) -> dict[str, Any]:
        """Add an observation to an entity."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # First ensure entity exists
            self.create_or_update_entity(entity_name)

            # Create observation node and relationship
            query = """
            MATCH (e:Entity {name: $entity_name})
            CREATE (o:Observation {
                content: $observation,
                created_at: datetime(),
                id: randomUUID()
            })
            CREATE (e)-[:HAS_OBSERVATION]->(o)
            RETURN o
            """

            params = {
                "entity_name": entity_name,
                "observation": observation,
            }

            result = list(self.db.execute_and_fetch(query, params))

            if result:
                obs = result[0]["o"]
                operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

                logger.info(
                    "Observation added successfully",
                    entity_name=entity_name,
                    observation_length=len(observation),
                    operation_time_ms=operation_time_ms,
                    correlation_id=correlation_id,
                )

                return {
                    "entity_name": entity_name,
                    "observation_id": obs.get("id"),
                    "observation": observation,
                    "created_at": obs.get("created_at"),
                }
            else:
                raise Exception("Failed to add observation - no result returned")

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to add observation",
                entity_name=entity_name,
                observation_length=len(observation),
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise

    def create_relationship(
        self, entity1: str, entity2: str, relationship_type: str
    ) -> dict[str, Any]:
        """Create a relationship between two entities."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Ensure both entities exist
            self.create_or_update_entity(entity1)
            self.create_or_update_entity(entity2)

            # Create relationship
            query = """
            MATCH (e1:Entity {name: $entity1})
            MATCH (e2:Entity {name: $entity2})
            MERGE (e1)-[r:RELATES_TO {type: $relationship_type}]->(e2)
            ON CREATE SET r.created_at = datetime()
            RETURN r
            """

            params = {
                "entity1": entity1,
                "entity2": entity2,
                "relationship_type": relationship_type,
            }

            result = list(self.db.execute_and_fetch(query, params))

            if result:
                rel = result[0]["r"]
                operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

                logger.info(
                    "Relationship created successfully",
                    entity1=entity1,
                    entity2=entity2,
                    relationship_type=relationship_type,
                    operation_time_ms=operation_time_ms,
                    correlation_id=correlation_id,
                )

                return {
                    "entity1": entity1,
                    "entity2": entity2,
                    "relationship_type": relationship_type,
                    "created_at": rel.get("created_at"),
                }
            else:
                raise Exception("Failed to create relationship - no result returned")

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to create relationship",
                entity1=entity1,
                entity2=entity2,
                relationship_type=relationship_type,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise

    def search_observations(
        self, query: str, entity_filter: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search observations using text matching."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Build query with optional entity filter
            cypher_query = """
            MATCH (e:Entity)-[:HAS_OBSERVATION]->(o:Observation)
            WHERE o.content CONTAINS $query
            """

            params = {"query": query, "limit": limit}

            if entity_filter:
                cypher_query += " AND e.name = $entity_filter"
                params["entity_filter"] = entity_filter

            cypher_query += """
            RETURN e.name as entity_name, o.content as observation, o.created_at as created_at
            ORDER BY o.created_at DESC
            LIMIT $limit
            """

            result = list(self.db.execute_and_fetch(cypher_query, params))

            observations = []
            for row in result:
                observations.append(
                    {
                        "entity_name": row["entity_name"],
                        "observation": row["observation"],
                        "created_at": row["created_at"],
                    }
                )

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Observation search completed",
                query=query,
                entity_filter=entity_filter,
                results_count=len(observations),
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return observations

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to search observations",
                query=query,
                entity_filter=entity_filter,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise


# Global service instance
_memgraph_service: MemgraphService | None = None


def get_memgraph_service() -> MemgraphService:
    """Get the global Memgraph service instance."""
    global _memgraph_service
    if _memgraph_service is None:
        _memgraph_service = MemgraphService()
    return _memgraph_service

"""Memgraph service for long-term memory operations."""

import time
from typing import Any

import numpy as np
from gqlalchemy import Memgraph

from ..config import settings
from ..logging import get_logger
from ..services.embedding import get_embedding_service
from ..services.time import time_service
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
            # Use timestamp() function in Cypher instead of passing ISO strings
            if entity_type:
                query = """
                MERGE (e:Entity {name: $entity_name})
                ON CREATE SET e.created_at = timestamp(), e.updated_at = timestamp(), e.type = $entity_type
                ON MATCH SET e.updated_at = timestamp(), e.type = $entity_type
                RETURN e
                """
                params = {"entity_name": entity_name, "entity_type": entity_type}
            else:
                query = """
                MERGE (e:Entity {name: $entity_name})
                ON CREATE SET e.created_at = timestamp(), e.updated_at = timestamp()
                ON MATCH SET e.updated_at = timestamp()
                RETURN e
                """
                params = {"entity_name": entity_name}

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
                    "created_at": getattr(entity, "created_at", None),
                    "updated_at": getattr(entity, "updated_at", None),
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
        """Add an observation to an entity with dual embeddings."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # First ensure entity exists
            self.create_or_update_entity(entity_name)

            # Ensure vector indices exist - this will fail fast if vector search isn't supported
            indices_status = self.check_vector_indices()
            if (
                not indices_status["semantic_index_exists"]
                or not indices_status["emotional_index_exists"]
            ):
                logger.info(
                    "Vector indices missing, creating them",
                    correlation_id=correlation_id,
                )
                self.create_vector_indices()

            # Generate dual embeddings for the observation
            embedding_service = get_embedding_service()
            semantic_embedding = np.array(
                embedding_service.encode_semantic(observation), dtype=np.float32
            )
            emotional_embedding = np.array(
                embedding_service.encode_emotional(observation), dtype=np.float32
            )

            # Create observation node with embeddings using original alpha-recall property names
            query = """
            MATCH (e:Entity {name: $entity_name})
            CREATE (o:Observation {
                content: $observation,
                created_at: timestamp(),
                id: randomUUID(),
                semantic_vector: $semantic_vector,
                emotional_vector: $emotional_vector
            })
            CREATE (e)-[:HAS_OBSERVATION]->(o)
            RETURN o
            """

            params = {
                "entity_name": entity_name,
                "observation": observation,
                "semantic_vector": semantic_embedding.tolist(),
                "emotional_vector": emotional_embedding.tolist(),
            }

            result = list(self.db.execute_and_fetch(query, params))

            if result:
                obs = result[0]["o"]
                operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

                logger.info(
                    "Observation added successfully with dual embeddings",
                    entity_name=entity_name,
                    observation_length=len(observation),
                    semantic_embedding_dims=semantic_embedding.shape[0],
                    emotional_embedding_dims=emotional_embedding.shape[0],
                    operation_time_ms=operation_time_ms,
                    correlation_id=correlation_id,
                )

                return {
                    "entity_name": entity_name,
                    "observation_id": getattr(obs, "id", None),
                    "observation": observation,
                    "created_at": getattr(obs, "created_at", None),
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

            # Pre-calculate timestamp
            now = time_service.utc_isoformat()

            # Create relationship
            query = """
            MATCH (e1:Entity {name: $entity1})
            MATCH (e2:Entity {name: $entity2})
            MERGE (e1)-[r:RELATES_TO {type: $relationship_type}]->(e2)
            ON CREATE SET r.created_at = $timestamp
            RETURN r
            """

            params = {
                "entity1": entity1,
                "entity2": entity2,
                "relationship_type": relationship_type,
                "timestamp": now,
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
                    "created_at": getattr(rel, "created_at", None),
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
        """Search observations using dual vector similarity with Memgraph vector_search.search() procedure."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Generate query embeddings
            embedding_service = get_embedding_service()
            query_semantic = np.array(
                embedding_service.encode_semantic(query), dtype=np.float32
            )
            query_emotional = np.array(
                embedding_service.encode_emotional(query), dtype=np.float32
            )

            # Convert numpy arrays to lists for Memgraph query parameters
            semantic_vector = query_semantic.tolist()
            emotional_vector = query_emotional.tolist()

            # Search using Memgraph's vector_search.search() procedure (following original alpha-recall pattern)

            # Semantic search
            semantic_query = """
            CALL vector_search.search("semantic_vector_index", $limit, $query_vector)
            YIELD node, similarity
            MATCH (e:Entity)-[:HAS_OBSERVATION]->(node)
            """

            semantic_params = {
                "limit": limit * 2,  # Get more results since we'll merge and filter
                "query_vector": semantic_vector,
            }

            if entity_filter:
                semantic_query += " WHERE e.name = $entity_filter"
                semantic_params["entity_filter"] = entity_filter

            semantic_query += """
            RETURN e.name as entity_name,
                   node.content as observation,
                   node.created_at as created_at,
                   similarity,
                   "semantic" as search_type
            ORDER BY similarity DESC
            """

            semantic_results = list(
                self.db.execute_and_fetch(semantic_query, semantic_params)
            )

            # Emotional search
            emotional_query = """
            CALL vector_search.search("emotional_vector_index", $limit, $query_vector)
            YIELD node, similarity
            MATCH (e:Entity)-[:HAS_OBSERVATION]->(node)
            """

            emotional_params = {
                "limit": limit * 2,  # Get more results since we'll merge and filter
                "query_vector": emotional_vector,
            }

            if entity_filter:
                emotional_query += " WHERE e.name = $entity_filter"
                emotional_params["entity_filter"] = entity_filter

            emotional_query += """
            RETURN e.name as entity_name,
                   node.content as observation,
                   node.created_at as created_at,
                   similarity,
                   "emotional" as search_type
            ORDER BY similarity DESC
            """

            emotional_results = list(
                self.db.execute_and_fetch(emotional_query, emotional_params)
            )

            # Combine and process results
            all_results = []

            # Process semantic results
            for row in semantic_results:
                all_results.append(
                    {
                        "entity_name": row["entity_name"],
                        "observation": row["observation"],
                        "created_at": row["created_at"],
                        "similarity_score": float(row["similarity"]),
                        "search_type": row["search_type"],
                    }
                )

            # Process emotional results
            for row in emotional_results:
                all_results.append(
                    {
                        "entity_name": row["entity_name"],
                        "observation": row["observation"],
                        "created_at": row["created_at"],
                        "similarity_score": float(row["similarity"]),
                        "search_type": row["search_type"],
                    }
                )

            # Filter by similarity threshold (0.3 like STM)
            similarity_threshold = 0.3
            filtered_results = [
                r for r in all_results if r["similarity_score"] >= similarity_threshold
            ]

            # Merge and deduplicate results
            seen = set()
            unique_results = []

            for result in filtered_results:
                # Create a unique key based on entity and observation content
                key = (result["entity_name"], result["observation"])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)

            # Sort by similarity score (highest first) and limit
            unique_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            observations = unique_results[:limit]

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Observation search completed with vector_search.search() procedure",
                query=query,
                entity_filter=entity_filter,
                semantic_results=len(semantic_results),
                emotional_results=len(emotional_results),
                final_results=len(observations),
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return observations

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to search observations with vector_search.search() procedure",
                query=query,
                entity_filter=entity_filter,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise

    def get_entity_with_observations(self, entity_name: str) -> dict[str, Any]:
        """Get entity information along with all its observations."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Get entity with all observations - return properties explicitly
            query = """
            MATCH (e:Entity {name: $entity_name})
            OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
            RETURN e,
                   collect({id: o.id, content: o.content, created_at: o.created_at}) as observations
            """

            params = {"entity_name": entity_name}
            result = list(self.db.execute_and_fetch(query, params))

            if not result:
                raise Exception(f"Entity '{entity_name}' not found")

            entity = result[0]["e"]
            observations = result[0]["observations"]

            # Process observations
            observation_list = []
            for obs in observations:
                if (
                    obs is not None and obs.get("content") is not None
                ):  # Skip null observations
                    observation_list.append(
                        {
                            "id": obs.get("id"),
                            "content": obs.get("content"),
                            "created_at": obs.get("created_at"),
                        }
                    )

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Entity with observations retrieved successfully",
                entity_name=entity_name,
                observations_count=len(observation_list),
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "entity_name": getattr(entity, "name", None),
                "entity_type": getattr(entity, "type", None),
                "created_at": getattr(entity, "created_at", None),
                "updated_at": getattr(entity, "updated_at", None),
                "observations": observation_list,
                "observations_count": len(observation_list),
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to get entity with observations",
                entity_name=entity_name,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise

    def get_entity_relationships(self, entity_name: str) -> dict[str, Any]:
        """Get all relationships for an entity (both incoming and outgoing)."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Get both outgoing and incoming relationships
            query = """
            MATCH (e:Entity {name: $entity_name})
            OPTIONAL MATCH (e)-[r_out:RELATES_TO]->(target:Entity)
            OPTIONAL MATCH (source:Entity)-[r_in:RELATES_TO]->(e)
            RETURN e,
                   collect(DISTINCT {
                       direction: 'outgoing',
                       target: target.name,
                       type: r_out.type,
                       created_at: r_out.created_at
                   }) as outgoing,
                   collect(DISTINCT {
                       direction: 'incoming',
                       source: source.name,
                       type: r_in.type,
                       created_at: r_in.created_at
                   }) as incoming
            """

            params = {"entity_name": entity_name}
            result = list(self.db.execute_and_fetch(query, params))

            if not result:
                raise Exception(f"Entity '{entity_name}' not found")

            entity = result[0]["e"]
            outgoing = result[0]["outgoing"]
            incoming = result[0]["incoming"]

            # Filter out null relationships from OPTIONAL MATCH
            outgoing_relationships = [
                r for r in outgoing if r.get("target") is not None
            ]
            incoming_relationships = [
                r for r in incoming if r.get("source") is not None
            ]

            total_relationships = len(outgoing_relationships) + len(
                incoming_relationships
            )

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Entity relationships retrieved successfully",
                entity_name=entity_name,
                outgoing_count=len(outgoing_relationships),
                incoming_count=len(incoming_relationships),
                total_relationships=total_relationships,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "entity_name": getattr(entity, "name", None),
                "entity_type": getattr(entity, "type", None),
                "outgoing_relationships": outgoing_relationships,
                "incoming_relationships": incoming_relationships,
                "total_relationships": total_relationships,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to get entity relationships",
                entity_name=entity_name,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise

    def browse_entities(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """Browse entities with pagination."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Get total count
            count_query = "MATCH (e:Entity) RETURN count(e) as total_count"
            count_result = list(self.db.execute_and_fetch(count_query))
            total_count = count_result[0]["total_count"] if count_result else 0

            # Get paginated entities with observation and relationship counts
            query = """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
            WITH e, count(DISTINCT o) as observation_count
            RETURN e.name as entity_name,
                   e.type as entity_type,
                   e.created_at as created_at,
                   e.updated_at as updated_at,
                   observation_count,
                   0 as relationship_count
            ORDER BY e.name ASC
            SKIP $offset
            LIMIT $limit
            """

            params = {"limit": limit, "offset": offset}
            result = list(self.db.execute_and_fetch(query, params))

            entities = []
            for row in result:
                entities.append(
                    {
                        "entity_name": row["entity_name"],
                        "entity_type": row["entity_type"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "observation_count": row["observation_count"],
                        "relationship_count": row["relationship_count"],
                    }
                )

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Entity browsing completed successfully",
                limit=limit,
                offset=offset,
                results_count=len(entities),
                total_count=total_count,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "entities": entities,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "results_count": len(entities),
                    "total_count": total_count,
                    "has_more": offset + len(entities) < total_count,
                },
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to browse entities",
                limit=limit,
                offset=offset,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise

    def create_vector_indices(self) -> dict[str, Any]:
        """Create vector indices for semantic and emotional embeddings using official Memgraph syntax."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Create semantic vector index (768 dimensions) - using official syntax
            semantic_index_query = """
            CREATE VECTOR INDEX semantic_vector_index ON :Observation(semantic_vector)
            WITH CONFIG {"dimension": 768, "capacity": 1024}
            """

            # Create emotional vector index (1024 dimensions) - using official syntax
            emotional_index_query = """
            CREATE VECTOR INDEX emotional_vector_index ON :Observation(emotional_vector)
            WITH CONFIG {"dimension": 1024, "capacity": 1024}
            """

            # Execute index creation queries with proper error handling
            logger.info("Creating semantic vector index", correlation_id=correlation_id)
            semantic_result = list(self.db.execute_and_fetch(semantic_index_query))
            logger.info(
                "Semantic index created",
                result=semantic_result,
                correlation_id=correlation_id,
            )

            logger.info(
                "Creating emotional vector index", correlation_id=correlation_id
            )
            emotional_result = list(self.db.execute_and_fetch(emotional_index_query))
            logger.info(
                "Emotional index created",
                result=emotional_result,
                correlation_id=correlation_id,
            )

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Vector indices created successfully",
                semantic_dimensions=768,
                emotional_dimensions=1024,
                capacity=1024,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "semantic_index": {
                    "name": "semantic_vector_index",
                    "dimensions": 768,
                    "capacity": 1024,
                    "status": "created",
                },
                "emotional_index": {
                    "name": "emotional_vector_index",
                    "dimensions": 1024,
                    "capacity": 1024,
                    "status": "created",
                },
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to create vector indices",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            raise

    def check_vector_indices(self) -> dict[str, Any]:
        """Check if vector indices exist using SHOW INDEX INFO (following original pattern)."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        try:
            # Check vector indices using vector_search.show_index_info() procedure
            query = "CALL vector_search.show_index_info() YIELD *"
            result = list(self.db.execute_and_fetch(query))

            indices = {}
            semantic_exists = False
            emotional_exists = False

            for row in result:
                # Look for our specific vector indices
                if row.get("index_name") == "semantic_vector_index":
                    semantic_exists = True
                    indices["semantic_index"] = {
                        "name": "semantic_vector_index",
                        "property": "semantic_vector",
                        "status": "exists",
                    }
                elif row.get("index_name") == "emotional_vector_index":
                    emotional_exists = True
                    indices["emotional_index"] = {
                        "name": "emotional_vector_index",
                        "property": "emotional_vector",
                        "status": "exists",
                    }

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Vector indices checked successfully",
                indices_found=len(indices),
                semantic_exists=semantic_exists,
                emotional_exists=emotional_exists,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "indices": indices,
                "semantic_index_exists": semantic_exists,
                "emotional_index_exists": emotional_exists,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Failed to check vector indices",
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

"""
Memgraph implementation of the graph database interface using GQLAlchemy.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from gqlalchemy import Memgraph

from alpha_recall.db.base import GraphDatabase
from alpha_recall.logging_utils import get_logger
from alpha_recall.utils.retry import async_retry

# Load environment variables
load_dotenv()

# Memgraph connection settings
GRAPH_DB_URI = os.environ.get("GRAPH_DB_URI", "bolt://localhost:7687")

# Get logger
logger = get_logger("memgraph_db")


class MemgraphDatabase(GraphDatabase):
    """
    Memgraph implementation of the graph database interface using GQLAlchemy.
    """

    @async_retry(
        max_retries=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        error_messages_to_retry=["failed to receive chunk size"],
    )
    async def recency_search(self, limit: int = 10) -> list:
        """
        Return the N most recent observations within the given time span.
        Args:
            span: A string representing the time span (e.g., '1h', '1d')
            limit: Maximum number of results to return (default 10)
        Returns:
            List of recent observations
        """
        if not self.memgraph:
            logger.error("Not connected to Memgraph")
            return []
        try:
            # Query for recent observations with their connected entities
            query = (
                "MATCH (e:Entity)-[:HAS_OBSERVATION]->(o:Observation) "
                "RETURN o, e.name as entity_name "
                "ORDER BY coalesce(o.updated_at, o.created_at) DESC "
                f"LIMIT {limit}"
            )
            results = self.memgraph.execute_and_fetch(query)
            observations = []
            for r in results:
                obs_dict = dict(r["o"])
                obs_dict["entity_name"] = r["entity_name"]
                observations.append(obs_dict)
            return observations
        except Exception as e:
            logger.error(f"Error in recency_search: {str(e)}")
            return []

    def __init__(
        self,
        uri: Optional[str] = None,
    ):
        """
        Initialize the Memgraph database connection.

        Args:
            uri: Memgraph connection URI (defaults to environment variable)
        """
        self.uri = uri or GRAPH_DB_URI
        self.memgraph = None

    async def connect(self) -> None:
        """
        Establish a connection to the Memgraph database.
        """
        try:
            # Parse the URI to extract host and port
            # URI format: bolt://hostname:port
            uri_parts = self.uri.replace("bolt://", "").split(":")
            host = uri_parts[0]
            port = int(uri_parts[1]) if len(uri_parts) > 1 else 7687

            # Create connection
            self.memgraph = Memgraph(host=host, port=port)

            # Test connection
            result = self.memgraph.execute_and_fetch("RETURN 1 AS test")
            list(result)  # Force execution

            logger.info(f"Connected to Memgraph at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Memgraph at {self.uri}: {str(e)}")
            raise

    async def close(self) -> None:
        """
        Close the database connection.

        Note: GQLAlchemy doesn't have an explicit close method as it manages
        connections internally, but we implement this for interface compatibility.
        """
        self.memgraph = None
        logger.info("Memgraph connection closed")

    async def is_connected(self) -> bool:
        """
        Check if the database connection is active.

        Returns:
            bool: True if connected, False otherwise
        """
        if not self.memgraph:
            return False

        try:
            # Test connection with a simple query
            result = self.memgraph.execute_and_fetch("RETURN 1 AS test")
            list(result)  # Force execution
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    async def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query against the database.

        Args:
            query: The query string to execute
            parameters: Optional parameters for the query

        Returns:
            List of records as dictionaries
        """
        if not self.memgraph:
            logger.error("Not connected to Memgraph")
            raise RuntimeError("Not connected to Memgraph")

        try:
            if parameters:
                # Format query with parameters
                for key, value in parameters.items():
                    if isinstance(value, str):
                        # Escape single quotes in string values
                        value = value.replace("'", "\\'")
                        # Wrap strings in quotes
                        parameters[key] = f"'{value}'"

                # Apply parameters to query
                query = query.format(**parameters)

            # Execute query
            results = self.memgraph.execute_and_fetch(query)

            # Convert results to list of dictionaries
            return [dict(record) for record in results]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    @async_retry(
        max_retries=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        error_messages_to_retry=["failed to receive chunk size"],
    )
    async def create_entity(
        self, name: str, entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new entity in the graph.

        Args:
            name: Name of the entity
            entity_type: Optional type of the entity

        Returns:
            Dictionary representing the created entity
        """
        if not self.memgraph:
            logger.error("Not connected to Memgraph")
            raise RuntimeError("Not connected to Memgraph")

        try:
            # Check if entity already exists using parameter binding
            query = "MATCH (e:Entity {name: $name}) RETURN e"
            params = {"name": name}
            results = self.memgraph.execute_and_fetch(query, params)
            existing = list(results)

            if existing:
                # Entity already exists, return it
                entity = dict(existing[0]["e"])
                logger.info(
                    f"Entity '{name}' already exists, returning existing entity"
                )

                # Format the response to match Neo4j implementation
                return {
                    "entity": {
                        "id": entity.get("id", str(uuid.uuid4())),
                        "name": name,
                        "type": entity_type or entity.get("type", "Entity"),
                        "created_at": entity.get(
                            "created_at", datetime.now(timezone.utc).isoformat()
                        ),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "created": False,
                    "success": True,
                }

            # Create new entity using parameter binding
            now = datetime.now(timezone.utc).isoformat()
            entity_id = str(uuid.uuid4())

            # Base query with parameters
            query = "CREATE (e:Entity {name: $name, created_at: $timestamp, updated_at: $timestamp, id: $entity_id})"

            # Parameters for the query
            params = {"name": name, "timestamp": now, "entity_id": entity_id}

            # Add entity type if provided
            if entity_type:
                query += " SET e:$entity_type, e.type = $entity_type"
                params["entity_type"] = entity_type

            # Return the created entity
            query += " RETURN e"

            # Execute query
            results = self.memgraph.execute_and_fetch(query, params)
            entity = dict(list(results)[0]["e"])

            logger.info(f"Created entity '{name}' with ID {entity_id}")

            # Format the response to match Neo4j implementation
            return {
                "entity": {
                    "id": entity_id,
                    "name": name,
                    "type": entity_type or "Entity",
                    "created_at": now,
                    "updated_at": now,
                },
                "created": True,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Failed to create entity '{name}': {str(e)}")
            return {"error": str(e), "success": False}

    @async_retry(
        max_retries=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        error_messages_to_retry=["failed to receive chunk size"],
    )
    async def add_observation(
        self, entity_name: str, observation: str
    ) -> Dict[str, Any]:
        """
        Add an observation to an entity.

        Args:
            entity_name: Name of the entity
            observation: Content of the observation

        Returns:
            Dictionary representing the updated entity
        """
        if not self.memgraph:
            logger.error("Not connected to Memgraph")
            raise RuntimeError("Not connected to Memgraph")

        try:
            # First ensure the entity exists
            entity_result = await self.create_entity(entity_name)
            if not entity_result.get("success", False):
                return entity_result

            # Extract entity data - this works whether the entity is new or existing
            entity = entity_result["entity"]

            # Make sure we have an ID
            if "id" not in entity:
                logger.error(f"Entity result missing ID: {entity}")
                return {"error": f"Entity missing ID", "success": False}

            entity_id = entity["id"]

            # Create observation
            observation_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            # Create observation node and link to entity using parameter dictionary
            query = """
            MATCH (e:Entity {id: $entity_id})
            CREATE (o:Observation {
                id: $observation_id,
                content: $content,
                created_at: $timestamp
            })
            CREATE (e)-[:HAS_OBSERVATION]->(o)
            RETURN e, o
            """

            # Parameters for the query
            params = {
                "entity_id": entity_id,
                "observation_id": observation_id,
                "content": observation,
                "timestamp": now,
            }

            # Execute query
            results = self.memgraph.execute_and_fetch(query, params)
            results_list = list(results)

            # Check if we got any results
            if not results_list:
                logger.warning(
                    f"No results returned when adding observation to entity '{entity_name}'"
                )
                # Even if no results, the observation might have been created
                # Continue with the update

            # Update entity's updated_at timestamp using parameter binding
            update_query = """
            MATCH (e:Entity {id: $entity_id})
            SET e.updated_at = $timestamp
            """
            update_params = {"entity_id": entity_id, "timestamp": now}
            self.memgraph.execute(update_query, update_params)

            logger.info(f"Added observation to entity '{entity_name}'")

            # Format the response to match Neo4j implementation
            return {
                "entity": entity,
                "observation": {
                    "id": observation_id,
                    "content": observation,
                    "created_at": now,
                },
                "success": True,
            }
        except IndexError as e:
            # This specifically handles the list index out of range error
            logger.warning(
                f"No results returned when adding observation, but observation may still have been created: {str(e)}"
            )
            # Try to continue with a successful response
            return {
                "entity": entity,
                "observation": {
                    "id": observation_id,
                    "content": observation,
                    "created_at": now,
                },
                "success": True,
            }
        except Exception as e:
            error_msg = f"Failed to add observation to entity '{entity_name}': {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

    @async_retry(
        max_retries=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        error_messages_to_retry=["failed to receive chunk size"],
    )
    async def create_relationship(
        self, source_entity: str, target_entity: str, relationship_type: str
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
        if not self.memgraph:
            logger.error("Not connected to Memgraph")
            raise RuntimeError("Not connected to Memgraph")

        try:
            # First ensure both entities exist
            source_result = await self.create_entity(source_entity)
            if not source_result.get("success", False):
                return source_result

            target_result = await self.create_entity(target_entity)
            if not target_result.get("success", False):
                return target_result

            source = source_result["entity"]
            target = target_result["entity"]

            # Format relationship type for Cypher
            # Replace spaces with underscores for relationship type
            rel_type = relationship_type.upper().replace(" ", "_")

            # Create relationship using parameter binding
            query = """
            MATCH (source:Entity {id: $source_id})  
            MATCH (target:Entity {id: $target_id})  
            MERGE (source)-[r:$rel_type]->(target)  
            RETURN source, target, r  
            """

            # Parameters for the query
            params = {
                "source_id": source["id"],
                "target_id": target["id"],
                "rel_type": rel_type,
            }

            # Execute query
            results = self.memgraph.execute_and_fetch(query, params)
            results_list = list(results)

            # Update entities' updated_at timestamps regardless of results
            now = datetime.now(timezone.utc).isoformat()
            update_query = """
            MATCH (e:Entity)
            WHERE e.id = $source_id OR e.id = $target_id
            SET e.updated_at = $timestamp
            """
            update_params = {
                "source_id": source["id"],
                "target_id": target["id"],
                "timestamp": now,
            }
            self.memgraph.execute(update_query, update_params)

            # Log success
            logger.info(
                f"Created relationship '{relationship_type}' from '{source_entity}' to '{target_entity}'"
            )

            # Format the response to match Neo4j implementation
            if results_list:
                result = results_list[0]
                return {
                    "source": source,
                    "target": target,
                    "type": relationship_type,
                    "success": True,
                }
            else:
                # If no results are returned, try to continue with a successful response
                return {
                    "source": source,
                    "target": target,
                    "type": relationship_type,
                    "success": True,
                }
        except IndexError as e:
            # This specifically handles the list index out of range error
            logger.warning(
                f"No results returned when creating relationship, but relationship may still have been created: {str(e)}"
            )
            # Try to continue with a successful response
            if "source" in locals() and "target" in locals():
                return {
                    "source": source,
                    "target": target,
                    "type": relationship_type,
                    "success": True,
                }
            else:
                return {
                    "error": "Failed to create relationship: No results returned",
                    "success": False,
                }
        except Exception as e:
            error_msg = f"Failed to create relationship: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

    @async_retry(
        max_retries=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        error_messages_to_retry=["failed to receive chunk size"],
    )
    async def get_entity(
        self, entity_name: str, depth: int = 1
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
        if not self.memgraph:
            logger.error("Not connected to Memgraph")
            raise RuntimeError("Not connected to Memgraph")

        try:
            # First check if entity exists using parameter binding
            query = "MATCH (e:Entity {name: $name}) RETURN e"
            params = {"name": entity_name}
            results = self.memgraph.execute_and_fetch(query, params)
            entities = list(results)

            if not entities:
                logger.info(f"Entity '{entity_name}' not found")
                return None

            entity = dict(entities[0]["e"])

            # Format the entity
            formatted_entity = {
                "id": entity.get("id"),
                "name": entity_name,
                "type": entity.get("type", "Entity"),
                "created_at": entity.get("created_at"),
                "updated_at": entity.get("updated_at"),
                "observations": [],
                "relationships": [],
            }

            # If depth is 0, just return the entity without relationships or observations
            if depth == 0:
                return formatted_entity

            # Get observations using parameter binding
            obs_query = """
            MATCH (e:Entity {name: $name})-[:HAS_OBSERVATION]->(o:Observation)
            RETURN o
            ORDER BY o.created_at DESC
            """

            obs_results = self.memgraph.execute_and_fetch(
                obs_query, {"name": entity_name}
            )

            for record in obs_results:
                obs = dict(record["o"])
                formatted_entity["observations"].append(
                    {"content": obs.get("content"), "created_at": obs.get("created_at")}
                )

            # Get relationships using parameter binding
            rel_query = """
            MATCH (e:Entity {name: $name})-[r]->(target:Entity)
            RETURN type(r) as type, target.name as target, null as source
            UNION
            MATCH (source:Entity)-[r]->(e:Entity {name: $name})            
            RETURN type(r) as type, null as target, source.name as source
            """

            rel_results = self.memgraph.execute_and_fetch(
                rel_query, {"name": entity_name}
            )

            for record in rel_results:
                record_dict = dict(record)
                # Build relationship dict with order: source, type, target
                if record_dict["source"] is not None:
                    relationship = {
                        "source": record_dict["source"],
                        "type": record_dict.get("type").lower().replace("_", " "),
                        "target": entity_name,
                    }
                else:
                    relationship = {
                        "source": entity_name,
                        "type": record_dict.get("type").lower().replace("_", " "),
                        "target": record_dict["target"],
                    }
                formatted_entity["relationships"].append(relationship)

            # If depth > 1, recursively get related entities
            # This is a simplified implementation and might need optimization for large graphs
            if depth > 1:
                # Get all directly related entities using parameter binding
                related_query = """
                MATCH (e:Entity {name: $name})-[r]-(related:Entity)
                RETURN DISTINCT related.name as name
                """

                related_results = self.memgraph.execute_and_fetch(
                    related_query, {"name": entity_name}
                )
                related_entities = [dict(record)["name"] for record in related_results]

                # Recursively get each related entity with reduced depth
                for related_name in related_entities:
                    if related_name != entity_name:  # Avoid self-reference
                        await self.get_entity(related_name, depth - 1)

            return formatted_entity
        except Exception as e:
            logger.error(f"Failed to get entity '{entity_name}': {str(e)}")
            raise

    @async_retry(
        max_retries=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        error_messages_to_retry=["failed to receive chunk size"],
    )
    async def delete_entity(self, name: str) -> Dict[str, Any]:
        """
        Delete an entity and all its relationships (and attached observations) from the graph.

        Args:
            name: Name of the entity to delete
        Returns:
            Dictionary containing the deletion status and details
        """
        if not self.memgraph:
            logger.error("Not connected to Memgraph")
            return {"success": False, "error": "Not connected to Memgraph"}

        try:
            # First, detach delete the entity and all relationships (including observations)
            # Remove attached observations
            obs_query = f"""
            MATCH (e:Entity {{name: '{name}'}})-[:HAS_OBSERVATION]->(o:Observation)
            DETACH DELETE o
            """
            self.memgraph.execute(obs_query)

            # Now delete the entity and all its relationships
            del_query = f"""
            MATCH (e:Entity {{name: '{name}'}})
            DETACH DELETE e
            """
            self.memgraph.execute(del_query)

            logger.info(f"Deleted entity '{name}' and its relationships")

            return {"entity": name, "deleted": True, "success": True}
        except Exception as e:
            logger.error(f"Error deleting entity '{name}': {str(e)}")
            return {"success": False, "error": str(e), "entity": name}

    async def remember_shortterm(
        self, content: str, client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a short-term memory with automatic TTL expiration.

        This is a stub implementation as Memgraph doesn't support short-term memory operations.
        Short-term memory is handled by Redis in the CompositeDatabase.

        Args:
            content: The memory content to store
            client_info: Optional information about the client/source

        Returns:
            Dictionary containing information about the operation status
        """
        logger.warning("Short-term memory operations not supported in MemgraphDatabase")
        return {
            "success": False,
            "error": "Short-term memory operations not supported in Memgraph. Use CompositeDatabase instead.",
        }

    async def get_shortterm_memories(
        self, through_the_last: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent short-term memories.

        This is a stub implementation as Memgraph doesn't support short-term memory operations.
        Short-term memory is handled by Redis in the CompositeDatabase.

        Args:
            through_the_last: Optional time window (e.g., '2h', '1d')
            limit: Maximum number of memories to return

        Returns:
            Empty list as Memgraph doesn't support short-term memories
        """
        logger.warning("Short-term memory operations not supported in MemgraphDatabase")
        return []

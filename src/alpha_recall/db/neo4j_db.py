"""
Neo4j implementation of the graph database interface.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from alpha_recall.db.base import GraphDatabase
from alpha_recall.logging_utils import get_logger

# Load environment variables
load_dotenv()

# Neo4j connection settings
GRAPH_DB_URI = os.environ.get("GRAPH_DB_URI", "bolt://localhost:7687")
GRAPH_DB_USER = os.environ.get("GRAPH_DB_USER", "neo4j")
GRAPH_DB_PASSWORD = os.environ.get("GRAPH_DB_PASSWORD", "")

# Get logger
logger = get_logger("neo4j_db")


class Neo4jDatabase(GraphDatabase):
    """
    Neo4j implementation of the graph database interface.
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_timeout: int = 30,
    ):
        """
        Initialize the Neo4j database connection.
        
        Args:
            uri: Neo4j connection URI (defaults to environment variable)
            user: Neo4j username (defaults to environment variable)
            password: Neo4j password (defaults to environment variable)
            max_connection_lifetime: Maximum lifetime of a connection in seconds
            max_connection_pool_size: Maximum size of the connection pool
            connection_timeout: Connection timeout in seconds
        """
        self.uri = uri or GRAPH_DB_URI
        self.user = user or GRAPH_DB_USER
        self.password = password or GRAPH_DB_PASSWORD
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_timeout = connection_timeout
        self.driver: Optional[AsyncDriver] = None
        
        logger.debug(f"Initialized Neo4j connection to {self.uri}")
    
    async def connect(self) -> None:
        """
        Establish a connection to the Neo4j database.
        
        Raises:
            ServiceUnavailable: If the Neo4j server is unavailable
            AuthError: If authentication fails
        """
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size,
                connection_timeout=self.connection_timeout,
            )
            # Verify connection
            await self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    async def close(self) -> None:
        """
        Close the Neo4j database connection.
        """
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")
    
    async def is_connected(self) -> bool:
        """
        Check if the Neo4j connection is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.driver:
            return False
        
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Connection check failed: {str(e)}")
            return False
    
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the Neo4j database.
        
        Args:
            query: Cypher query string
            parameters: Optional parameters for the query
            
        Returns:
            List of records as dictionaries
            
        Raises:
            Exception: If the query fails or connection is not established
        """
        if not self.driver:
            raise Exception("Not connected to Neo4j")
        
        parameters = parameters or {}
        results = []
        
        try:
            async with self.driver.session() as session:
                result = await session.run(query, parameters)
                async for record in result:
                    # Convert Neo4j record to dictionary
                    results.append(dict(record))
                
                # Consume the result to ensure all records are fetched
                await result.consume()
                
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.debug(f"Query: {query}, Parameters: {parameters}")
            raise
    
    async def create_entity(
        self, 
        name: str, 
        entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new entity in the Neo4j graph.
        
        Args:
            name: Name of the entity
            entity_type: Optional type of the entity
            
        Returns:
            Dictionary representing the created entity
        """
        # Default entity type if not provided
        entity_type = entity_type or "Entity"
        
        # Current timestamp
        timestamp = datetime.now().isoformat()
        
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET 
            e.type = $type,
            e.created_at = $timestamp
        ON MATCH SET 
            e.type = $type,
            e.updated_at = $timestamp
        RETURN e
        """
        
        parameters = {
            "name": name,
            "type": entity_type,
            "timestamp": timestamp
        }
        
        results = await self.execute_query(query, parameters)
        if not results:
            raise Exception(f"Failed to create entity: {name}")
        
        return results[0]["e"]
    
    async def add_observation(
        self, 
        entity_name: str, 
        observation: str
    ) -> Dict[str, Any]:
        """
        Add an observation to an entity in the Neo4j graph.
        
        Args:
            entity_name: Name of the entity
            observation: Content of the observation
            
        Returns:
            Dictionary representing the updated entity with the new observation
        """
        # Current timestamp
        timestamp = datetime.now().isoformat()
        
        # First ensure the entity exists
        await self.create_entity(entity_name)
        
        # Then add the observation
        query = """
        MATCH (e:Entity {name: $entity_name})
        CREATE (o:Observation {content: $content, created_at: $timestamp})
        CREATE (e)-[:HAS_OBSERVATION]->(o)
        RETURN e, o
        """
        
        parameters = {
            "entity_name": entity_name,
            "content": observation,
            "timestamp": timestamp
        }
        
        results = await self.execute_query(query, parameters)
        if not results:
            raise Exception(f"Failed to add observation to entity: {entity_name}")
        
        # Return the entity with the new observation
        return {
            "entity": results[0]["e"],
            "observation": results[0]["o"]
        }
    
    async def create_relationship(
        self, 
        source_entity: str, 
        target_entity: str, 
        relationship_type: str
    ) -> Dict[str, Any]:
        """
        Create a relationship between two entities in the Neo4j graph.
        
        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relationship_type: Type of the relationship
            
        Returns:
            Dictionary representing the created relationship
        """
        # Ensure both entities exist
        await self.create_entity(source_entity)
        await self.create_entity(target_entity)
        
        # Current timestamp
        timestamp = datetime.now().isoformat()
        
        # Create the relationship
        query = """
        MATCH (source:Entity {name: $source_name})
        MATCH (target:Entity {name: $target_name})
        MERGE (source)-[r:`$rel_type`]->(target)
        ON CREATE SET r.created_at = $timestamp
        ON MATCH SET r.updated_at = $timestamp
        RETURN source, target, r
        """
        
        # Neo4j doesn't allow parameterized relationship types,
        # so we need to replace the placeholder manually
        query = query.replace("`$rel_type`", f"`{relationship_type}`")
        
        parameters = {
            "source_name": source_entity,
            "target_name": target_entity,
            "timestamp": timestamp
        }
        
        results = await self.execute_query(query, parameters)
        if not results:
            raise Exception(
                f"Failed to create relationship from {source_entity} "
                f"to {target_entity} as {relationship_type}"
            )
        
        return {
            "source": results[0]["source"],
            "target": results[0]["target"],
            "relationship": results[0]["r"]
        }
    
    async def get_entity(
        self, 
        entity_name: str, 
        depth: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entity and its relationships from the Neo4j graph.
        
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
        # Validate depth
        if depth < 0:
            depth = 0
        
        if depth == 0:
            # Just get the entity and its observations
            query = """
            MATCH (e:Entity {name: $name})
            OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
            RETURN e, collect(o) as observations
            """
            
            parameters = {"name": entity_name}
            
        else:
            # Get the entity, its observations, and relationships up to the specified depth
            # This is a more complex query that uses variable-length path matching
            # Note: We need to use string formatting for the path length since Neo4j doesn't allow
            # parameter substitution in relationship patterns
            query = f"""
            MATCH (e:Entity {{name: $name}})
            OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
            WITH e, collect(DISTINCT o) as observations
            OPTIONAL MATCH path = (e)-[r*1..{depth}]-(related)
            WHERE related:Entity
            RETURN 
                e, 
                observations,
                collect(DISTINCT [type(r[-1]), startNode(r[-1]).name, endNode(r[-1]).name]) as relationships
            """
            
            parameters = {
                "name": entity_name
                # depth is now directly in the query string
            }
        
        results = await self.execute_query(query, parameters)
        if not results or not results[0]["e"]:
            return None
        
        # Process the results into a structured response
        entity_data = dict(results[0]["e"])
        entity_data["observations"] = [dict(o) for o in results[0]["observations"]]
        
        if depth > 0:
            # Process relationships
            relationships = []
            for rel_data in results[0]["relationships"]:
                # Each relationship is a list with [type, source_name, target_name]
                if len(rel_data) == 3:
                    rel_type = rel_data[0]
                    source_name = rel_data[1]
                    target_name = rel_data[2]
                    
                    # Create a structured relationship object
                    relationship = {
                        "type": rel_type,
                        "source": source_name,
                        "target": target_name
                    }
                    
                    # Only add if it's not already in the list
                    if relationship not in relationships:
                        relationships.append(relationship)
            
            entity_data["relationships"] = relationships
        
        return entity_data

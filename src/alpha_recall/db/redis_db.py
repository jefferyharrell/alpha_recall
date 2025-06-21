"""
Redis database for short-term memory storage.

This module provides a Redis-based implementation for storing and retrieving
short-term memories with automatic TTL expiration.
"""

import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np
import redis.asyncio as redis

from alpha_recall.logging_utils import get_logger

logger = get_logger(__name__)

# Default TTL values
DEFAULT_TEST_TTL = 120  # 2 minutes for testing
DEFAULT_PROD_TTL = 259200  # 72 hours (3 days) for production

# Default embedding configuration
DEFAULT_EMBEDDING_SERVER_URL = "http://localhost:6004/api/v1/embeddings/semantic"
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_VECTOR_SIZE = 768  # Dimension of all-mpnet-base-v2 embeddings

# Default emotional embedding configuration
DEFAULT_EMOTIONAL_EMBEDDING_URL = "http://localhost:6004/api/v1/embeddings/emotion"
DEFAULT_EMOTIONAL_EMBEDDING_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_EMOTIONAL_VECTOR_SIZE = 1024  # Dimension of emotion embeddings


class RedisShortTermMemory:
    """
    Redis-based implementation for short-term memory storage with TTL.

    This class provides methods for storing and retrieving short-term memories
    with automatic time-based expiration.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        ttl: int = DEFAULT_PROD_TTL,
        key_prefix: str = "alpha:stm:",
        embedding_server_url: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        emotional_embedding_url: Optional[str] = None,
        emotional_embedding_model: str = DEFAULT_EMOTIONAL_EMBEDDING_MODEL,
    ):
        """
        Initialize the Redis short-term memory database.

        Args:
            host: Redis server hostname
            port: Redis server port
            password: Optional Redis password
            db: Redis database number
            ttl: Time-to-live in seconds for stored memories
            key_prefix: Prefix for Redis keys
            embedding_server_url: URL of the embedding server API
            embedding_model: Name of the embedding model to use
            emotional_embedding_url: URL of the emotional embedding server API
            emotional_embedding_model: Name of the emotional embedding model to use
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.client = None
        
        # Embedding configuration
        self.embedding_server_url = embedding_server_url or os.environ.get(
            "EMBEDDING_SERVER_URL", DEFAULT_EMBEDDING_SERVER_URL
        )
        self.embedding_model = embedding_model
        self.vector_size = DEFAULT_VECTOR_SIZE
        
        # Emotional embedding configuration
        self.emotional_embedding_url = emotional_embedding_url or os.environ.get(
            "EMOTIONAL_EMBEDDING_URL", DEFAULT_EMOTIONAL_EMBEDDING_URL
        )
        self.emotional_embedding_model = emotional_embedding_model
        self.emotional_vector_size = DEFAULT_EMOTIONAL_VECTOR_SIZE
        
        self._index_created = False
        self._emotional_index_created = False

    async def connect(self) -> None:
        """
        Establish a connection to the Redis server.
        """
        try:
            logger.info(f"Connecting to Redis at {self.host}:{self.port}")
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=False,  # We need binary support for vectors
            )
            # Test connection
            await self.client.ping()
            logger.info("Successfully connected to Redis")
            
            # Create vector indices if they don't exist
            await self._ensure_vector_index()
            await self._ensure_emotional_index_exists()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    async def close(self) -> None:
        """
        Close the Redis connection.
        """
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

    async def is_connected(self) -> bool:
        """
        Check if the Redis connection is active.

        Returns:
            bool: True if connected, False otherwise
        """
        if not self.client:
            return False

        try:
            return await self.client.ping()
        except Exception:
            return False

    def _generate_key(self) -> str:
        """
        Generate a unique key for a short-term memory entry.

        Returns:
            str: A unique key in the format prefix:timestamp-uuid
        """
        timestamp = int(time.time() * 1000)  # Milliseconds since epoch
        random_suffix = str(uuid.uuid4())[:8]  # First 8 chars of a UUID
        return f"{self.key_prefix}{timestamp}-{random_suffix}"

    def _parse_duration(self, duration_str: str) -> Optional[timedelta]:
        """
        Parse a duration string into a timedelta object.

        Args:
            duration_str: A string like '2h', '1d', etc.

        Returns:
            Optional[timedelta]: The parsed duration or None if invalid
        """
        if not duration_str:
            return None

        try:
            # Extract the number and unit
            unit = duration_str[-1].lower()
            value = int(duration_str[:-1])

            if unit == "m":
                return timedelta(minutes=value)
            elif unit == "h":
                return timedelta(hours=value)
            elif unit == "d":
                return timedelta(days=value)
            else:
                logger.warning(f"Invalid duration unit: {unit}")
                return None
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse duration '{duration_str}': {str(e)}")
            return None

    async def _ensure_vector_index(self) -> None:
        """
        Ensure the vector search index exists in Redis.
        """
        if self._index_created:
            return
            
        try:
            # Check if index already exists
            try:
                await self.client.execute_command('FT.INFO', 'idx:stm')
                self._index_created = True
                logger.info("Vector index 'idx:stm' already exists")
                return
            except redis.ResponseError:
                # Index doesn't exist, create it
                pass
            
            # Create the index for vector search (semantic embeddings)
            # Keep using 'embedding' for compatibility with existing memories
            await self.client.execute_command(
                'FT.CREATE', 'idx:stm',
                'ON', 'HASH',
                'PREFIX', '1', self.key_prefix,
                'SCHEMA',
                'content', 'TEXT',
                'embedding', 'VECTOR', 'FLAT', '6',
                'TYPE', 'FLOAT32',
                'DIM', str(self.vector_size),
                'DISTANCE_METRIC', 'COSINE'
            )
            self._index_created = True
            logger.info("Created vector index 'idx:stm' for short-term memories")
        except Exception as e:
            logger.warning(f"Failed to create vector index: {str(e)}")
            # Continue without vector search capability

    async def _ensure_emotional_index_exists(self) -> None:
        """
        Ensure the emotional vector search index exists for short-term memories.
        """
        if self._emotional_index_created:
            return
            
        try:
            # Check if index already exists
            try:
                await self.client.execute_command('FT.INFO', 'idx:stm_emotional')
                self._emotional_index_created = True
                logger.info("Emotional vector index 'idx:stm_emotional' already exists")
                return
            except redis.ResponseError:
                # Index doesn't exist, create it
                pass
            
            # Create the index for emotional vector search
            await self.client.execute_command(
                'FT.CREATE', 'idx:stm_emotional',
                'ON', 'HASH',
                'PREFIX', '1', self.key_prefix,
                'SCHEMA',
                'content', 'TEXT',
                'embedding_emotional', 'VECTOR', 'FLAT', '6',
                'TYPE', 'FLOAT32',
                'DIM', str(self.emotional_vector_size),
                'DISTANCE_METRIC', 'COSINE'
            )
            self._emotional_index_created = True
            logger.info("Created emotional vector index 'idx:stm_emotional' for short-term memories")
        except Exception as e:
            logger.warning(f"Failed to create emotional vector index: {str(e)}")
            # Continue without emotional vector search capability
            
    async def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embeddings for the given text using the HTTP embedding server.
        
        Args:
            text: Text to embed
            
        Returns:
            Embeddings as a numpy array or None if embedding fails
        """
        try:
            payload = {"texts": [text]}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.embedding_server_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0,
                )
                
                response.raise_for_status()
                data = response.json()
                
                embeddings = data.get("embeddings", None)
                if not embeddings or not isinstance(embeddings, list):
                    logger.error(f"Embedding server returned unexpected data: {data}")
                    return None
                    
                return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {str(e)}")
            return None

    async def _embed_text_emotional(self, text: str) -> Optional[np.ndarray]:
        """
        Generate emotional embeddings for the given text using the HTTP emotional embedding server.
        
        Args:
            text: Text to embed emotionally
            
        Returns:
            Emotional embeddings as a numpy array or None if embedding fails
        """
        try:
            payload = {"texts": [text]}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.emotional_embedding_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0,
                )
                
                response.raise_for_status()
                data = response.json()
                
                embeddings = data.get("embeddings", None)
                if not embeddings or not isinstance(embeddings, list):
                    logger.error(f"Emotional embedding server returned unexpected data: {data}")
                    return None
                    
                return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to generate emotional embedding: {str(e)}")
            return None

    async def store_memory(
        self, content: str, client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a short-term memory with automatic TTL and optional embedding.

        Args:
            content: The memory content to store
            client_info: Optional information about the client

        Returns:
            Dict[str, Any]: Information about the stored memory
        """
        if not await self.is_connected():
            await self.connect()

        # Create memory object
        timestamp = datetime.utcnow().isoformat()
        
        # Generate a unique key
        key = self._generate_key()
        
        # Prepare hash data
        hash_data = {
            "content": content,
            "created_at": timestamp,
            "client": json.dumps(client_info or {}),
        }
        
        # Try to generate semantic embedding
        embedding = await self._embed_text(content)
        if embedding is not None:
            # Store semantic embedding as binary data (use 'embedding' for compatibility)
            hash_data["embedding"] = embedding.tobytes()
        
        # Try to generate emotional embedding
        emotional_embedding = await self._embed_text_emotional(content)
        if emotional_embedding is not None:
            # Store emotional embedding as binary data
            hash_data["embedding_emotional"] = emotional_embedding.tobytes()
        
        # Store in Redis as hash with TTL
        await self.client.hset(key, mapping=hash_data)
        await self.client.expire(key, self.ttl)

        # Return memory info
        return {
            "id": key,
            "content": content,
            "created_at": timestamp,
            "ttl": self.ttl,
            "client": client_info or {},
            "has_semantic_embedding": embedding is not None,
            "has_emotional_embedding": emotional_embedding is not None,
        }

    async def get_recent_memories(
        self, through_the_last: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent short-term memories.

        Args:
            through_the_last: Optional time window (e.g., '2h', '1d')
            limit: Maximum number of memories to return

        Returns:
            List[Dict[str, Any]]: List of recent memories, newest first
        """
        if not await self.is_connected():
            await self.connect()

        # Get all keys matching the prefix
        keys = await self.client.keys(f"{self.key_prefix}*")

        # Convert bytes to strings if necessary
        if keys and isinstance(keys[0], bytes):
            keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in keys]

        # Sort keys by timestamp (newest first)
        keys.sort(reverse=True)

        # Apply time window filter if specified
        if through_the_last:
            cutoff_time = None
            duration = self._parse_duration(through_the_last)

            if duration:
                cutoff_time = datetime.utcnow() - duration
                filtered_keys = []

                for key in keys:
                    # Get memory data from hash
                    memory_data = await self.client.hgetall(key)
                    if memory_data:
                        # Decode bytes if necessary
                        if b'created_at' in memory_data:
                            created_at_str = memory_data[b'created_at'].decode('utf-8')
                        else:
                            created_at_str = memory_data.get('created_at', '')
                            
                        if created_at_str:
                            created_at = datetime.fromisoformat(created_at_str)
                            
                            # Keep only memories newer than the cutoff time
                            if created_at >= cutoff_time:
                                filtered_keys.append(key)

                            # Stop once we have enough keys
                            if len(filtered_keys) >= limit:
                                break

                keys = filtered_keys[:limit]
            else:
                # If duration parsing failed, just take the most recent ones
                keys = keys[:limit]
        else:
            # No time filter, just take the most recent ones
            keys = keys[:limit]

        # Retrieve memory data for each key
        memories = []
        for key in keys:
            memory_data = await self.client.hgetall(key)
            if memory_data:
                # Convert hash data to dictionary
                memory = {}
                for field, value in memory_data.items():
                    field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                    if field_str == 'content':
                        memory['content'] = value.decode('utf-8') if isinstance(value, bytes) else value
                    elif field_str == 'created_at':
                        memory['created_at'] = value.decode('utf-8') if isinstance(value, bytes) else value
                    elif field_str == 'client':
                        client_str = value.decode('utf-8') if isinstance(value, bytes) else value
                        memory['client'] = json.loads(client_str) if client_str else {}
                    # Skip embedding field in response
                    
                memory["id"] = key
                memories.append(memory)

        return memories

    async def semantic_search_memories(
        self, 
        query: str, 
        limit: int = 10,
        through_the_last: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on short-term memories using vector similarity.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return (default: 10)
            through_the_last: Optional time window filter (e.g., '2h', '1d')
            
        Returns:
            List[Dict[str, Any]]: List of semantically similar memories with scores
        """
        if not await self.is_connected():
            await self.connect()
            
        # Generate embedding for the query
        query_embedding = await self._embed_text(query)
        if query_embedding is None:
            logger.warning("Failed to generate query embedding, falling back to recent memories")
            return await self.get_recent_memories(through_the_last, limit)
            
        try:
            # Prepare the vector search query
            # For time filtering, we need to combine with a TAG filter
            # Use 'embedding' for compatibility with existing memories
            base_query = f"*=>[KNN {limit} @embedding $vec_param AS score]"
            
            # Execute vector search
            results = await self.client.execute_command(
                'FT.SEARCH', 'idx:stm',
                base_query,
                'PARAMS', '2', 
                'vec_param', query_embedding.tobytes(),
                'SORTBY', 'score', 'DESC',
                'RETURN', '4', 'content', 'created_at', 'client', 'score',
                'DIALECT', '2'
            )
            
            # Parse search results
            memories = []
            if results and len(results) > 1:
                num_results = results[0]
                
                # Results come in format: [num_results, key1, [field1, value1, ...], key2, ...]
                for i in range(1, len(results), 2):
                    if i + 1 < len(results):
                        key = results[i]
                        fields = results[i + 1]
                        
                        # Convert key to string if bytes
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        
                        # Parse fields into dictionary
                        memory = {"id": key_str}
                        for j in range(0, len(fields), 2):
                            if j + 1 < len(fields):
                                field = fields[j]
                                value = fields[j + 1]
                                
                                field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                                
                                if field_str == 'content':
                                    memory['content'] = value.decode('utf-8') if isinstance(value, bytes) else value
                                elif field_str == 'created_at':
                                    memory['created_at'] = value.decode('utf-8') if isinstance(value, bytes) else value
                                elif field_str == 'client':
                                    client_str = value.decode('utf-8') if isinstance(value, bytes) else value
                                    memory['client'] = json.loads(client_str) if client_str else {}
                                elif field_str == 'score':
                                    # Convert similarity score to float
                                    score_str = value.decode('utf-8') if isinstance(value, bytes) else value
                                    memory['similarity_score'] = float(score_str)
                        
                        # Apply time filter if specified
                        if through_the_last and 'created_at' in memory:
                            duration = self._parse_duration(through_the_last)
                            if duration:
                                cutoff_time = datetime.utcnow() - duration
                                created_at = datetime.fromisoformat(memory['created_at'])
                                if created_at < cutoff_time:
                                    continue  # Skip this memory
                        
                        memories.append(memory)
                        
                        # Stop if we have enough results
                        if len(memories) >= limit:
                            break
            
            return memories
            
        except redis.ResponseError as e:
            logger.warning(f"Vector search failed: {str(e)}, falling back to recent memories")
            # Fallback to time-based retrieval if vector search fails
            return await self.get_recent_memories(through_the_last, limit)

    async def emotional_search_memories(
        self, 
        query: str, 
        limit: int = 10,
        through_the_last: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform emotional search on short-term memories using emotional similarity.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return (default: 10)
            through_the_last: Optional time window filter (e.g., '2h', '1d')
            
        Returns:
            List[Dict[str, Any]]: List of emotionally similar memories with scores
        """
        if not await self.is_connected():
            await self.connect()
            
        # Generate emotional embedding for the query
        query_embedding = await self._embed_text_emotional(query)
        if query_embedding is None:
            logger.warning("Failed to generate query emotional embedding, falling back to recent memories")
            return await self.get_recent_memories(through_the_last, limit)
            
        try:
            # Prepare the emotional vector search query
            base_query = f"*=>[KNN {limit} @embedding_emotional $vec_param AS score]"
            
            # Execute emotional vector search
            results = await self.client.execute_command(
                'FT.SEARCH', 'idx:stm_emotional',
                base_query,
                'PARAMS', '2', 
                'vec_param', query_embedding.tobytes(),
                'SORTBY', 'score', 'DESC',
                'RETURN', '4', 'content', 'created_at', 'client', 'score',
                'DIALECT', '2'
            )
            
            # Parse search results (same parsing logic as semantic search)
            memories = []
            if results and len(results) > 1:
                num_results = results[0]
                
                # Results come in format: [num_results, key1, [field1, value1, ...], key2, ...]
                for i in range(1, len(results), 2):
                    if i + 1 < len(results):
                        key = results[i]
                        fields = results[i + 1]
                        
                        # Convert key to string if bytes
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        
                        # Parse fields into dictionary
                        memory = {"id": key_str}
                        for j in range(0, len(fields), 2):
                            if j + 1 < len(fields):
                                field = fields[j]
                                value = fields[j + 1]
                                
                                field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                                
                                if field_str == 'content':
                                    memory['content'] = value.decode('utf-8') if isinstance(value, bytes) else value
                                elif field_str == 'created_at':
                                    memory['created_at'] = value.decode('utf-8') if isinstance(value, bytes) else value
                                elif field_str == 'client':
                                    client_str = value.decode('utf-8') if isinstance(value, bytes) else value
                                    memory['client'] = json.loads(client_str) if client_str else {}
                                elif field_str == 'score':
                                    # Convert similarity score to float
                                    score_str = value.decode('utf-8') if isinstance(value, bytes) else value
                                    memory['emotional_score'] = float(score_str)
                        
                        # Apply time filter if specified
                        if through_the_last and 'created_at' in memory:
                            duration = self._parse_duration(through_the_last)
                            if duration:
                                cutoff_time = datetime.utcnow() - duration
                                created_at = datetime.fromisoformat(memory['created_at'])
                                if created_at < cutoff_time:
                                    continue  # Skip this memory
                        
                        memories.append(memory)
                        
                        # Stop if we have enough results
                        if len(memories) >= limit:
                            break
            
            return memories
            
        except redis.ResponseError as e:
            logger.warning(f"Emotional vector search failed: {str(e)}, falling back to recent memories")
            # Fallback to time-based retrieval if emotional vector search fails
            return await self.get_recent_memories(through_the_last, limit)

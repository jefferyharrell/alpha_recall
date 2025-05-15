"""
Redis database for short-term memory storage.

This module provides a Redis-based implementation for storing and retrieving
short-term memories with automatic TTL expiration.
"""

import json
import time
import uuid
import redis.asyncio as redis
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from alpha_recall.logging_utils import get_logger

logger = get_logger(__name__)

# Default TTL values
DEFAULT_TEST_TTL = 120  # 2 minutes for testing
DEFAULT_PROD_TTL = 259200  # 72 hours (3 days) for production


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
        key_prefix: str = "alpha:stm:"
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
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.client = None
        
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
                decode_responses=True
            )
            # Test connection
            await self.client.ping()
            logger.info("Successfully connected to Redis")
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
            
            if unit == 'm':
                return timedelta(minutes=value)
            elif unit == 'h':
                return timedelta(hours=value)
            elif unit == 'd':
                return timedelta(days=value)
            else:
                logger.warning(f"Invalid duration unit: {unit}")
                return None
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse duration '{duration_str}': {str(e)}")
            return None
    
    async def store_memory(
        self,
        content: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a short-term memory with automatic TTL.
        
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
        memory = {
            "content": content,
            "created_at": timestamp,
            "client": client_info or {}
        }
        
        # Generate a unique key
        key = self._generate_key()
        
        # Store in Redis with TTL
        await self.client.set(key, json.dumps(memory))
        await self.client.expire(key, self.ttl)
        
        # Return memory info
        return {
            "id": key,
            "content": content,
            "created_at": timestamp,
            "ttl": self.ttl,
            "client": client_info or {}
        }
    
    async def get_recent_memories(
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
            List[Dict[str, Any]]: List of recent memories, newest first
        """
        if not await self.is_connected():
            await self.connect()
        
        # Get all keys matching the prefix
        keys = await self.client.keys(f"{self.key_prefix}*")
        
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
                    # Get memory data
                    memory_json = await self.client.get(key)
                    if memory_json:
                        memory = json.loads(memory_json)
                        created_at = datetime.fromisoformat(memory["created_at"])
                        
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
            memory_json = await self.client.get(key)
            if memory_json:
                memory = json.loads(memory_json)
                memory["id"] = key
                memories.append(memory)
        
        return memories

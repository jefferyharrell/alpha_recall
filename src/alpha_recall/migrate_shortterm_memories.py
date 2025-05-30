#!/usr/bin/env python3
"""
Migration script to add embeddings to existing short-term memories in Redis.

This script:
1. Finds all existing short-term memories stored as strings
2. Converts them to hash format with embeddings
3. Preserves TTL values
4. Removes old string entries after successful migration
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import redis.asyncio as redis
from dotenv import load_dotenv

# Add parent directory to path to import from alpha_recall
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_recall.db.redis_db import RedisShortTermMemory
from alpha_recall.logging_utils import get_logger

load_dotenv()

logger = get_logger(__name__)


async def migrate_memories():
    """Migrate existing short-term memories to include embeddings."""
    
    # Initialize Redis connection
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    redis_password = os.environ.get("REDIS_PASSWORD")
    redis_db = int(os.environ.get("REDIS_DB", 0))
    
    # Create Redis short-term memory instance
    stm = RedisShortTermMemory(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        db=redis_db
    )
    
    try:
        await stm.connect()
        logger.info("Connected to Redis for migration")
        
        # Get all short-term memory keys
        pattern = f"{stm.key_prefix}*"
        keys = await stm.client.keys(pattern)
        
        if not keys:
            logger.info("No short-term memories found to migrate")
            return
        
        # Convert bytes to strings if necessary
        if keys and isinstance(keys[0], bytes):
            keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in keys]
        
        logger.info(f"Found {len(keys)} short-term memories to check")
        
        migrated_count = 0
        skipped_count = 0
        failed_count = 0
        
        for key in keys:
            try:
                # Check if this is already a hash (new format)
                key_type = await stm.client.type(key)
                if isinstance(key_type, bytes):
                    key_type = key_type.decode('utf-8')
                
                if key_type == 'hash':
                    logger.debug(f"Key {key} is already in hash format, skipping")
                    skipped_count += 1
                    continue
                
                # Get the string data (old format)
                memory_json = await stm.client.get(key)
                if not memory_json:
                    logger.warning(f"Key {key} has no data, skipping")
                    skipped_count += 1
                    continue
                
                # Parse the JSON
                if isinstance(memory_json, bytes):
                    memory_json = memory_json.decode('utf-8')
                memory = json.loads(memory_json)
                
                # Extract fields
                content = memory.get('content', '')
                created_at = memory.get('created_at', datetime.utcnow().isoformat())
                client_info = memory.get('client', {})
                
                # Get current TTL
                ttl = await stm.client.ttl(key)
                if ttl <= 0:
                    logger.warning(f"Key {key} has no TTL or has expired, skipping")
                    skipped_count += 1
                    continue
                
                # Generate embedding
                logger.info(f"Generating embedding for memory: {key}")
                embedding = await stm._embed_text(content)
                
                # Prepare hash data
                hash_data = {
                    "content": content,
                    "created_at": created_at,
                    "client": json.dumps(client_info),
                }
                
                if embedding is not None:
                    hash_data["embedding"] = embedding.tobytes()
                    logger.info(f"Successfully generated embedding for {key}")
                else:
                    logger.warning(f"Failed to generate embedding for {key}, storing without embedding")
                
                # Delete the old string entry
                await stm.client.delete(key)
                
                # Store as hash with same key
                await stm.client.hset(key, mapping=hash_data)
                
                # Restore TTL
                await stm.client.expire(key, ttl)
                
                migrated_count += 1
                logger.info(f"Migrated memory {key} (TTL: {ttl}s)")
                
            except Exception as e:
                logger.error(f"Failed to migrate key {key}: {str(e)}")
                failed_count += 1
                continue
        
        # Create or update the vector index
        await stm._ensure_vector_index()
        
        logger.info(f"\nMigration complete!")
        logger.info(f"  Migrated: {migrated_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Total: {len(keys)}")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise
    finally:
        await stm.close()


async def verify_migration():
    """Verify that memories can be searched after migration."""
    
    # Initialize Redis connection
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    redis_password = os.environ.get("REDIS_PASSWORD")
    redis_db = int(os.environ.get("REDIS_DB", 0))
    
    stm = RedisShortTermMemory(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        db=redis_db
    )
    
    try:
        await stm.connect()
        
        # Try a test search
        logger.info("\nTesting semantic search...")
        results = await stm.semantic_search_memories("test", limit=5)
        
        if results:
            logger.info(f"Semantic search working! Found {len(results)} results")
            for i, result in enumerate(results[:3]):
                logger.info(f"  {i+1}. {result.get('content', '')[:50]}...")
                if 'similarity_score' in result:
                    logger.info(f"     Score: {result['similarity_score']}")
        else:
            logger.warning("No results found in test search")
            
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
    finally:
        await stm.close()


async def main():
    """Run the migration."""
    logger.info("Starting short-term memory migration...")
    logger.info("This will add embeddings to existing memories to enable semantic search")
    
    await migrate_memories()
    await verify_migration()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Migration script to add emotional embeddings to existing short-term memories in Redis.

This script:
1. Finds all existing short-term memories that have semantic embeddings but no emotional embeddings
2. Generates emotional embeddings for their content
3. Adds emotional embeddings to the existing hash records
4. Preserves all other fields and TTL values
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import redis.asyncio as redis
import numpy as np
from dotenv import load_dotenv

# Add parent directory to path to import from alpha_recall
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_recall.db.redis_db import RedisShortTermMemory
from alpha_recall.logging_utils import get_logger

load_dotenv()

logger = get_logger(__name__)


async def migrate_emotional_embeddings():
    """Add emotional embeddings to existing short-term memories."""
    
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
        logger.info("Connected to Redis")
        
        # Find all short-term memory keys
        pattern = f"{stm.key_prefix}*"
        keys = await stm.client.keys(pattern)
        
        if not keys:
            logger.info("No short-term memories found")
            return
        
        logger.info(f"Found {len(keys)} short-term memory keys")
        
        migrated_count = 0
        skipped_count = 0
        error_count = 0
        
        for key in keys:
            try:
                # Get the memory hash
                memory_data = await stm.client.hgetall(key)
                
                if not memory_data:
                    logger.warning(f"Empty memory data for key: {key}")
                    continue
                
                # Convert bytes keys to strings, handle values appropriately
                memory = {}
                for k, v in memory_data.items():
                    k_str = k.decode('utf-8') if isinstance(k, bytes) else k
                    
                    # Only decode text fields to UTF-8, keep binary fields as bytes
                    if k_str in ['content', 'created_at', 'client']:
                        v_str = v.decode('utf-8') if isinstance(v, bytes) else v
                        memory[k_str] = v_str
                    else:
                        # Keep binary fields (like embeddings) as raw bytes
                        memory[k_str] = v
                
                # Check if this memory already has emotional embedding
                if 'embedding_emotional' in memory:
                    logger.debug(f"Memory {key} already has emotional embedding, skipping")
                    skipped_count += 1
                    continue
                
                # Check if this memory has content to generate emotional embedding from
                content = memory.get('content')
                if not content:
                    logger.warning(f"Memory {key} has no content, skipping")
                    skipped_count += 1
                    continue
                
                # Check if this memory has any embeddings at all (some might have no embeddings)
                has_semantic = 'embedding_semantic' in memory or 'embedding' in memory
                if not has_semantic:
                    logger.debug(f"Memory {key} has no semantic embedding, skipping emotional migration")
                    skipped_count += 1
                    continue
                
                # Generate emotional embedding
                logger.info(f"Generating emotional embedding for memory: {key}")
                emotional_embedding = await stm._embed_text_emotional(content)
                
                if emotional_embedding is not None:
                    # Add emotional embedding to the hash
                    await stm.client.hset(key, "embedding_emotional", emotional_embedding.tobytes())
                    logger.info(f"Successfully added emotional embedding to memory: {key}")
                    migrated_count += 1
                else:
                    logger.warning(f"Failed to generate emotional embedding for memory: {key}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing memory {key}: {str(e)}")
                error_count += 1
                continue
        
        # Ensure emotional index exists
        logger.info("Ensuring emotional vector index exists...")
        await stm._ensure_emotional_index_exists()
        
        logger.info(f"Migration completed:")
        logger.info(f"  - Migrated: {migrated_count}")
        logger.info(f"  - Skipped: {skipped_count}")
        logger.info(f"  - Errors: {error_count}")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise
    finally:
        await stm.close()


async def main():
    """Main function for the migration."""
    try:
        await migrate_emotional_embeddings()
        logger.info("Emotional embedding migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
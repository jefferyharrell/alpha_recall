#!/usr/bin/env python3
"""
Migration script to upgrade emotional vectors from 7-dimensional to 1024-dimensional.

This script:
1. Connects to Redis
2. Finds all short-term memories with 7-dimensional emotional embeddings
3. Re-generates 1024-dimensional emotional embeddings using the new service
4. Updates the records with new embeddings
5. Recreates the Redis emotional vector index with the new dimension

Usage:
    uv run python migrate_emotional_vectors.py [--dry-run]
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import argparse

import numpy as np
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from alpha_recall.db.redis_db import RedisShortTermMemory
from alpha_recall.logging_utils import configure_logging, get_logger

# Load environment variables
load_dotenv()

# Configure logging
configure_logging()
logger = get_logger("migrate_emotional_vectors")


async def migrate_emotional_vectors(dry_run: bool = False):
    """
    Migrate emotional vectors from 7-dimensional to 1024-dimensional.
    
    Args:
        dry_run: If True, only analyze existing data without making changes
    """
    logger.info("Starting emotional vector migration...")
    
    # Initialize Redis connection
    redis_stm = RedisShortTermMemory(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD"),
        db=int(os.getenv("REDIS_DB", "0")),
        emotional_embedding_url=os.getenv("EMOTIONAL_EMBEDDING_URL", "http://localhost:6004/sentiment-embeddings")
    )
    
    try:
        # Connect to Redis
        await redis_stm.connect()
        logger.info("Connected to Redis")
        
        # Get all short-term memory keys
        pattern = redis_stm.key_prefix + "*"
        keys = await redis_stm.client.keys(pattern)
        
        logger.info(f"Found {len(keys)} short-term memory entries")
        
        memories_with_old_vectors = []
        memories_without_emotional = []
        
        # Analyze existing memories
        for key in keys:
            try:
                memory_data = await redis_stm.client.hgetall(key)
                if not memory_data:
                    continue
                
                # Decode the data, handling binary fields
                decoded_data = {}
                for field, value in memory_data.items():
                    field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                    
                    # Handle binary embedding fields
                    if field_str in ['embedding', 'embedding_emotional', 'embedding_emotional_binary']:
                        if isinstance(value, bytes):
                            # For JSON-encoded embeddings, try to decode as UTF-8
                            if field_str in ['embedding', 'embedding_emotional']:
                                try:
                                    value_str = value.decode('utf-8')
                                except UnicodeDecodeError:
                                    # Skip this field if it can't be decoded
                                    logger.warning(f"Skipping binary field {field_str} for key {key}")
                                    continue
                            else:
                                # For binary embedding data, skip decoding
                                continue
                        else:
                            value_str = value
                    else:
                        # For other fields, decode normally
                        try:
                            value_str = value.decode('utf-8') if isinstance(value, bytes) else value
                        except UnicodeDecodeError:
                            logger.warning(f"Failed to decode field {field_str} for key {key}, skipping")
                            continue
                    
                    decoded_data[field_str] = value_str
                
                # Check if this memory has emotional embedding
                if 'embedding_emotional' in decoded_data:
                    try:
                        # Parse the emotional embedding (JSON format)
                        embedding_data = json.loads(decoded_data['embedding_emotional'])
                        if isinstance(embedding_data, list):
                            embedding_dim = len(embedding_data)
                            if embedding_dim == 7:
                                memories_with_old_vectors.append({
                                    'key': key.decode('utf-8') if isinstance(key, bytes) else key,
                                    'content': decoded_data.get('content', ''),
                                    'old_embedding_dim': embedding_dim
                                })
                            elif embedding_dim == 1024:
                                logger.debug(f"Memory {key} already has 1024-dimensional emotional embedding")
                            else:
                                logger.warning(f"Memory {key} has unexpected emotional embedding dimension: {embedding_dim}")
                    except (json.JSONDecodeError, KeyError):
                        logger.debug(f"Memory {key} has non-JSON emotional embedding (likely binary)")
                        # For binary embeddings, we need to check the dimension differently
                        # We'll need to regenerate all binary embeddings since we can't easily determine their dimension
                        memories_with_old_vectors.append({
                            'key': key.decode('utf-8') if isinstance(key, bytes) else key,
                            'content': decoded_data.get('content', ''),
                            'old_embedding_dim': 'binary_unknown'
                        })
                elif 'embedding_emotional_binary' in memory_data:
                    # This memory has binary emotional embedding but no JSON version
                    # We need to regenerate it
                    memories_with_old_vectors.append({
                        'key': key.decode('utf-8') if isinstance(key, bytes) else key,
                        'content': decoded_data.get('content', ''),
                        'old_embedding_dim': 'binary_only'
                    })
                else:
                    memories_without_emotional.append({
                        'key': key.decode('utf-8') if isinstance(key, bytes) else key,
                        'content': decoded_data.get('content', '')
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to analyze memory {key}: {str(e)}")
                continue
        
        logger.info(f"Analysis complete:")
        logger.info(f"  - Memories with 7-dimensional emotional vectors: {len(memories_with_old_vectors)}")
        logger.info(f"  - Memories without emotional vectors: {len(memories_without_emotional)}")
        
        if dry_run:
            logger.info("DRY RUN: Would migrate the following memories:")
            for memory in memories_with_old_vectors[:5]:  # Show first 5 as examples
                logger.info(f"  - Key: {memory['key']}, Content: {memory['content'][:50]}...")
            if len(memories_with_old_vectors) > 5:
                logger.info(f"  - ... and {len(memories_with_old_vectors) - 5} more")
            return
        
        # Perform the migration
        if memories_with_old_vectors:
            logger.info("Starting migration of emotional vectors...")
            
            # Drop and recreate the emotional vector index
            try:
                await redis_stm.client.execute_command('FT.DROPINDEX', 'idx:stm_emotional')
                logger.info("Dropped old emotional vector index")
            except Exception as e:
                logger.info(f"No existing emotional index to drop: {str(e)}")
            
            # Create new index with 1024 dimensions
            await redis_stm._create_emotional_vector_index()
            logger.info("Created new emotional vector index with 1024 dimensions")
            
            successful_migrations = 0
            failed_migrations = 0
            
            for memory in memories_with_old_vectors:
                try:
                    # Generate new 1024-dimensional emotional embedding
                    new_embedding = await redis_stm._embed_text_emotional(memory['content'])
                    
                    if new_embedding is not None:
                        # Update the memory with new embedding
                        embedding_bytes = new_embedding.astype(np.float32).tobytes()
                        await redis_stm.client.hset(
                            memory['key'],
                            'embedding_emotional',
                            json.dumps(new_embedding.tolist())
                        )
                        await redis_stm.client.hset(
                            memory['key'],
                            'embedding_emotional_binary',
                            embedding_bytes
                        )
                        
                        successful_migrations += 1
                        logger.debug(f"Migrated emotional vector for {memory['key']}")
                    else:
                        logger.warning(f"Failed to generate new emotional embedding for {memory['key']}")
                        failed_migrations += 1
                        
                except Exception as e:
                    logger.error(f"Failed to migrate {memory['key']}: {str(e)}")
                    failed_migrations += 1
            
            logger.info(f"Migration complete:")
            logger.info(f"  - Successfully migrated: {successful_migrations}")
            logger.info(f"  - Failed migrations: {failed_migrations}")
        
        # Optionally generate emotional embeddings for memories that don't have them
        if memories_without_emotional:
            logger.info(f"Found {len(memories_without_emotional)} memories without emotional embeddings")
            
            # Ask user if they want to generate emotional embeddings for these
            if not dry_run:
                response = input("Would you like to generate emotional embeddings for memories that don't have them? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    logger.info("Generating emotional embeddings for memories without them...")
                    
                    successful_generations = 0
                    failed_generations = 0
                    
                    for memory in memories_without_emotional:
                        try:
                            # Generate emotional embedding
                            emotional_embedding = await redis_stm._embed_text_emotional(memory['content'])
                            
                            if emotional_embedding is not None:
                                # Update the memory with emotional embedding
                                embedding_bytes = emotional_embedding.astype(np.float32).tobytes()
                                await redis_stm.client.hset(
                                    memory['key'],
                                    'embedding_emotional',
                                    json.dumps(emotional_embedding.tolist())
                                )
                                await redis_stm.client.hset(
                                    memory['key'],
                                    'embedding_emotional_binary',
                                    embedding_bytes
                                )
                                
                                successful_generations += 1
                                logger.debug(f"Generated emotional embedding for {memory['key']}")
                            else:
                                logger.warning(f"Failed to generate emotional embedding for {memory['key']}")
                                failed_generations += 1
                                
                        except Exception as e:
                            logger.error(f"Failed to generate emotional embedding for {memory['key']}: {str(e)}")
                            failed_generations += 1
                    
                    logger.info(f"Emotional embedding generation complete:")
                    logger.info(f"  - Successfully generated: {successful_generations}")
                    logger.info(f"  - Failed generations: {failed_generations}")
    
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise
    
    finally:
        # Close Redis connection
        if hasattr(redis_stm, 'client') and redis_stm.client:
            await redis_stm.client.close()
            logger.info("Closed Redis connection")


async def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(description="Migrate emotional vectors from 7D to 1024D")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze existing data without making changes"
    )
    
    args = parser.parse_args()
    
    try:
        await migrate_emotional_vectors(dry_run=args.dry_run)
        logger.info("Migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
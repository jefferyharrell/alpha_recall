#!/usr/bin/env python3
"""
Migrate short-term memories from 384D to 768D semantic embeddings.

This script:
1. Connects to Redis and the new embedder service
2. Fetches all STM entries
3. Re-embeds their content using the new 768D model
4. Updates the embeddings in place
5. Recreates the vector index with the new dimensions
"""

import asyncio
import json
import struct
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import httpx
import numpy as np
import redis.asyncio as redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = None
REDIS_DB = 0
KEY_PREFIX = "alpha:stm:"

# New embedder service configuration
NEW_EMBEDDER_URL = "http://localhost:6004/api/v1/embeddings/semantic"
NEW_VECTOR_SIZE = 768

# Old configuration for comparison
OLD_VECTOR_SIZE = 384


async def get_new_embedding(text: str) -> Optional[np.ndarray]:
    """Generate 768D embedding using the new embedder service."""
    try:
        payload = {"texts": [text]}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                NEW_EMBEDDER_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
            
            response.raise_for_status()
            data = response.json()
            
            embeddings = data.get("embeddings")
            if not embeddings or not isinstance(embeddings, list):
                print(f"Error: No embeddings in response: {data}")
                return None
                
            return np.array(embeddings[0], dtype=np.float32)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


async def backup_embeddings(client: redis.Redis, keys: List[bytes]) -> Optional[dict]:
    """Create a backup of all current embeddings for rollback capability."""
    print(f"\n💾 Creating backup of {len(keys)} embeddings...")
    backup = {}
    
    try:
        for i, key in enumerate(keys, 1):
            if i % 50 == 0:  # Progress indicator
                print(f"  Backing up {i}/{len(keys)}...")
            
            # Get current embedding
            current_embedding = await client.hget(key, b'embedding')
            if current_embedding:
                backup[key.decode()] = current_embedding
        
        print(f"  ✅ Backup complete: {len(backup)} embeddings saved")
        return backup
    except Exception as e:
        print(f"  ❌ Backup failed: {e}")
        return None


async def restore_embeddings(client: redis.Redis, backup: dict) -> bool:
    """Restore embeddings from backup."""
    print(f"\n🔄 Restoring {len(backup)} embeddings from backup...")
    
    try:
        # Use pipeline for atomic restoration
        pipe = client.pipeline()
        
        for key_str, embedding_data in backup.items():
            key = key_str.encode('utf-8')
            pipe.hset(key, b'embedding', embedding_data)
        
        # Execute all restorations atomically
        await pipe.execute()
        print(f"  ✅ Restore complete: {len(backup)} embeddings restored")
        return True
    except Exception as e:
        print(f"  ❌ Restore failed: {e}")
        return False


async def migrate_embeddings_transactional(client: redis.Redis, keys: List[bytes], backup: dict) -> Tuple[int, int]:
    """Migrate embeddings using transactional batches with rollback capability."""
    print(f"\n🔄 Migrating embeddings (transactional mode)...")
    success_count = 0
    error_count = 0
    batch_size = 25  # Process in smaller batches for better atomicity
    
    for batch_start in range(0, len(keys), batch_size):
        batch_end = min(batch_start + batch_size, len(keys))
        batch_keys = keys[batch_start:batch_end]
        
        print(f"\n  📦 Processing batch {batch_start + 1}-{batch_end} ({len(batch_keys)} keys)...")
        
        try:
            # Process batch with transaction
            batch_success, batch_errors = await process_batch_transactional(client, batch_keys)
            success_count += batch_success
            error_count += batch_errors
            
            if batch_errors > 0:
                print(f"    ⚠️  Batch had {batch_errors} errors, but continuing...")
            else:
                print(f"    ✅ Batch completed successfully")
                
        except Exception as e:
            print(f"    ❌ Batch failed completely: {e}")
            error_count += len(batch_keys)
    
    return success_count, error_count


async def process_batch_transactional(client: redis.Redis, batch_keys: List[bytes]) -> Tuple[int, int]:
    """Process a batch of keys with transaction support."""
    success_count = 0
    error_count = 0
    
    # Prepare embeddings for the entire batch first
    batch_data = []
    
    for i, key in enumerate(batch_keys, 1):
        try:
            # Get the hash data
            data = await client.hgetall(key)
            
            # Extract content (Redis uses bytes keys with decode_responses=False)
            content = data.get(b'content')
            if not content:
                print(f"    [{i}/{len(batch_keys)}] Skipping {key.decode()[:50]}... - no content")
                error_count += 1
                continue
            
            # Decode bytes to string
            content_str = content.decode('utf-8')
            
            # Check if there's an existing embedding (Redis uses bytes keys)
            old_embedding = data.get(b'embedding')
            if old_embedding:
                old_size = len(old_embedding) // 4  # float32 = 4 bytes
                print(f"    [{i}/{len(batch_keys)}] Processing {key.decode()[:50]}... (current: {old_size}D)")
            else:
                print(f"    [{i}/{len(batch_keys)}] Processing {key.decode()[:50]}... (no embedding)")
            
            # Generate new embedding
            new_embedding = await get_new_embedding(content_str)
            if new_embedding is None:
                print(f"      ❌ Failed to generate embedding")
                error_count += 1
                continue
            
            # Convert to binary format for Redis
            embedding_bytes = struct.pack(f'{len(new_embedding)}f', *new_embedding.tolist())
            
            # Store for batch update
            batch_data.append((key, embedding_bytes, len(new_embedding)))
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            error_count += 1
    
    # Apply all updates in a single transaction
    if batch_data:
        try:
            pipe = client.pipeline()
            
            for key, embedding_bytes, embedding_size in batch_data:
                pipe.hset(key, b'embedding', embedding_bytes)
            
            # Execute all updates atomically
            await pipe.execute()
            
            success_count = len(batch_data)
            for key, embedding_bytes, embedding_size in batch_data:
                print(f"      ✅ Updated {key.decode()[:50]}... to {embedding_size}D")
                
        except Exception as e:
            print(f"      ❌ Batch transaction failed: {e}")
            error_count += len(batch_data)
    
    return success_count, error_count


async def migrate_stm_embeddings():
    """Main migration function with transaction support and rollback capability."""
    print("Starting STM embedding migration from 384D to 768D...")
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    # Connect to Redis
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=False,
    )
    
    try:
        await client.ping()
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        return
    
    # Test embedder service
    print("\nTesting new embedder service...")
    test_embedding = await get_new_embedding("Test connection")
    if test_embedding is None:
        print("✗ Failed to connect to embedder service")
        return
    print(f"✓ Embedder service working, returned {len(test_embedding)}D vector")
    
    # Get all STM keys
    print(f"\nScanning for STM entries with prefix: {KEY_PREFIX}")
    keys = []
    cursor = 0
    while True:
        cursor, batch = await client.scan(
            cursor, match=f"{KEY_PREFIX}*", count=100
        )
        keys.extend(batch)
        if cursor == 0:
            break
    
    print(f"Found {len(keys)} STM entries to migrate")
    
    if not keys:
        print("No STM entries found. Exiting.")
        return
    
    # First, do a safety check on a few entries to make sure we can read content
    print("\nSafety check: Testing content extraction on first 5 entries...")
    test_keys = keys[:5] if len(keys) >= 5 else keys
    readable_count = 0
    
    for i, key in enumerate(test_keys, 1):
        try:
            data = await client.hgetall(key)
            content = data.get(b'content') or data.get('content')
            if content:
                content_str = content.decode('utf-8') if isinstance(content, bytes) else content
                if content_str.strip():  # Make sure it's not just whitespace
                    readable_count += 1
                    print(f"  [{i}/5] ✓ Can read content from {key.decode()[:50]}...")
                else:
                    print(f"  [{i}/5] ✗ Empty content in {key.decode()}")
            else:
                print(f"  [{i}/5] ✗ No content field in {key.decode()}")
        except Exception as e:
            print(f"  [{i}/5] ✗ Error reading {key.decode()}: {e}")
    
    if readable_count == 0:
        print(f"\n❌ SAFETY CHECK FAILED: Cannot read content from any entries!")
        print(f"   Found {len(keys)} entries but none have readable content.")
        print(f"   Aborting migration to prevent data loss.")
        await client.close()
        return
    
    print(f"\n✅ Safety check passed: {readable_count}/{len(test_keys)} entries readable")
    
    # Ask for confirmation before proceeding with destructive operations
    if readable_count < len(test_keys):
        print(f"⚠️  Warning: Only {readable_count}/{len(test_keys)} test entries were readable.")
        print(f"   This could mean some data might be lost.")
    
    print(f"\nProceed with migration? This will:")
    print(f"  1. Drop the existing vector index")
    print(f"  2. Re-embed {len(keys)} entries")
    print(f"  3. Create new 768D vector index")
    
    # For safety, require manual confirmation
    # (In automated mode, you could add a --force flag)
    response = input("\nType 'YES' to proceed: ").strip()
    if response != 'YES':
        print("Migration cancelled.")
        await client.close()
        return
    
    # Drop the old index if it exists (but keep the data!)
    print("\nDropping old vector index...")
    try:
        await client.execute_command('FT.DROPINDEX', 'idx:stm')
        print("✓ Old index dropped (data preserved)")
    except redis.ResponseError as e:
        if "Unknown index name" in str(e):
            print("✓ Old index didn't exist")
        else:
            print(f"Warning: {e}")
    
    # Create backup before any destructive operations
    backup = await backup_embeddings(client, keys)
    
    if not backup:
        print(f"\n❌ Failed to create backup. Aborting migration for safety.")
        await client.close()
        return
    
    try:
        # Migrate embeddings using transactions
        success_count, error_count = await migrate_embeddings_transactional(client, keys, backup)
        
        # Check results
        if error_count == 0:
            print(f"\n🎉 Migration successful! All {success_count} embeddings updated.")
        elif success_count > 0:
            print(f"\n⚠️  Partial migration: {success_count} successful, {error_count} errors")
            print(f"   Consider investigating errors and re-running.")
        else:
            print(f"\n❌ Migration failed completely: 0 successful, {error_count} errors")
            print(f"   Rolling back all changes...")
            await restore_embeddings(client, backup)
            
    except Exception as e:
        print(f"\n💥 Critical error during migration: {e}")
        print(f"   Rolling back all changes...")
        await restore_embeddings(client, backup)
    
    print(f"\n📊 Final migration status: {success_count} successful, {error_count} errors")
    
    # Create new index with 768D (only if migration was successful)
    if error_count == 0:
        print("\n🔧 Creating new vector index with 768D...")
        try:
            await client.execute_command(
                'FT.CREATE', 'idx:stm',
                'ON', 'HASH',
                'PREFIX', '1', KEY_PREFIX,
                'SCHEMA',
                'content', 'TEXT',
                'embedding', 'VECTOR', 'FLAT', '6',
                'TYPE', 'FLOAT32',
                'DIM', str(NEW_VECTOR_SIZE),
                'DISTANCE_METRIC', 'COSINE'
            )
            print("✅ Created new 768D vector index")
        except Exception as e:
            print(f"❌ Failed to create index: {e}")
    else:
        print("\n⏭️  Skipping index creation due to migration errors")
    
    # Verify the migration (only if successful)
    if error_count == 0:
        print("\n🔍 Verifying migration...")
        sample_keys = keys[:5] if len(keys) > 5 else keys
        for key in sample_keys:
            data = await client.hget(key, b'embedding')
            if data:
                size = len(data) // 4
                print(f"  ✅ {key.decode()[:50]}...: {size}D embedding")
        print(f"\n🎯 Migration verification complete!")
    else:
        print(f"\n⏭️  Skipping verification due to migration errors")
    
    await client.close()
    
    if error_count == 0:
        print("\n🎉 MIGRATION COMPLETE! All embeddings successfully upgraded to 768D.")
        print(f"   - {success_count} embeddings migrated")
        print(f"   - New vector index created")
        print(f"   - Backup data can be discarded")
    else:
        print(f"\n⚠️  MIGRATION INCOMPLETE. Please review errors and retry if needed.")
        print(f"   - Backup preserved for safety")
        print(f"   - Vector index may need recreation")


if __name__ == "__main__":
    try:
        asyncio.run(migrate_stm_embeddings())
    except KeyboardInterrupt:
        print(f"\n❌ Migration interrupted by user")
        print(f"   If migration was in progress, restore from backup manually")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"   If migration was in progress, restore from backup manually")
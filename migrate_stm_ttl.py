#!/usr/bin/env python3
"""
Migration script to extend existing short-term memory TTLs to 2 megasecond baseline.

This script:
1. Scans all existing alpha:stm:* keys
2. Calculates their current age based on remaining TTL
3. Extends their TTL to what it would have been if they'd started with 2,000,000 seconds
4. Preserves the natural expiration order while extending the timeline

Usage: uv run python migrate_stm_ttl.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import redis.asyncio as redis


# Constants
OLD_TTL_SECONDS = 259200  # 72 hours (3 days)
NEW_TTL_SECONDS = 2000000  # 2 megaseconds (~23 days)
STM_KEY_PATTERN = "alpha:stm:*"


async def get_redis_client():
    """Create and return a Redis client."""
    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD'),
        decode_responses=False  # Don't decode to handle binary data properly
    )


async def scan_stm_keys(client: redis.Redis) -> List[str]:
    """Scan and return all short-term memory keys."""
    keys = []
    async for key in client.scan_iter(match=STM_KEY_PATTERN.encode()):
        # Keys come back as bytes, decode them to strings
        keys.append(key.decode() if isinstance(key, bytes) else key)
    return keys


async def get_key_info(client: redis.Redis, key: str) -> Tuple[int, Dict]:
    """Get TTL and content for a key."""
    ttl = await client.ttl(key.encode())
    raw_content = await client.hgetall(key.encode())
    
    # Decode hash content from bytes to strings
    content = {}
    for k, v in raw_content.items():
        try:
            decoded_key = k.decode() if isinstance(k, bytes) else k
            decoded_value = v.decode() if isinstance(v, bytes) else v
            content[decoded_key] = decoded_value
        except UnicodeDecodeError:
            # Skip fields that can't be decoded
            pass
    
    return ttl, content


async def calculate_new_ttl(current_ttl: int) -> int:
    """
    Calculate the new TTL based on the retroactive extension logic.
    
    Logic:
    - If TTL is -1 (no expiration), set to NEW_TTL_SECONDS
    - If TTL is -2 (key doesn't exist), skip
    - Otherwise: age = OLD_TTL_SECONDS - current_ttl
    - new_ttl = NEW_TTL_SECONDS - age
    """
    if current_ttl == -1:
        # No expiration set, give it the full new TTL
        return NEW_TTL_SECONDS
    elif current_ttl == -2:
        # Key doesn't exist, skip
        return -2
    else:
        # Calculate age and extend retroactively
        age = OLD_TTL_SECONDS - current_ttl
        new_ttl = NEW_TTL_SECONDS - age
        
        # Ensure we don't set negative TTL
        return max(new_ttl, 60)  # Minimum 1 minute


async def migrate_stm_ttls():
    """Main migration function."""
    client = await get_redis_client()
    
    try:
        # Scan for all STM keys
        print("Scanning for short-term memory keys...")
        keys = await scan_stm_keys(client)
        print(f"Found {len(keys)} short-term memory keys")
        
        if not keys:
            print("No keys found to migrate")
            return
        
        # Process each key
        migration_stats = {
            'processed': 0,
            'skipped': 0,
            'extended': 0,
            'errors': 0
        }
        
        print("\nProcessing keys...")
        for key in keys:
            try:
                current_ttl, content = await get_key_info(client, key)
                
                if current_ttl == -2:
                    print(f"Key {key} doesn't exist, skipping")
                    migration_stats['skipped'] += 1
                    continue
                
                new_ttl = await calculate_new_ttl(current_ttl)
                
                if new_ttl == -2:
                    migration_stats['skipped'] += 1
                    continue
                
                # Apply the new TTL
                await client.expire(key.encode(), new_ttl)
                
                # Parse timestamp from content for logging
                timestamp = "unknown"
                if 'timestamp' in content:
                    try:
                        ts = datetime.fromisoformat(content['timestamp'].replace('Z', '+00:00'))
                        timestamp = ts.strftime('%Y-%m-%d %H:%M:%S UTC')
                    except:
                        pass
                
                print(f"Extended {key}: {current_ttl}s → {new_ttl}s (created: {timestamp})")
                migration_stats['processed'] += 1
                migration_stats['extended'] += 1
                
            except Exception as e:
                print(f"Error processing {key}: {e}")
                migration_stats['errors'] += 1
        
        # Summary
        print(f"\nMigration complete!")
        print(f"Keys processed: {migration_stats['processed']}")
        print(f"Keys extended: {migration_stats['extended']}")
        print(f"Keys skipped: {migration_stats['skipped']}")
        print(f"Errors: {migration_stats['errors']}")
        
        if migration_stats['extended'] > 0:
            print(f"\n✅ Successfully extended {migration_stats['extended']} memories")
            print(f"Old TTL baseline: {OLD_TTL_SECONDS:,} seconds ({OLD_TTL_SECONDS/3600:.1f} hours)")
            print(f"New TTL baseline: {NEW_TTL_SECONDS:,} seconds ({NEW_TTL_SECONDS/86400:.1f} days)")
        
    finally:
        await client.aclose()


if __name__ == "__main__":
    print("Alpha Short-Term Memory TTL Migration Script")
    print("=" * 50)
    print(f"Extending TTLs from {OLD_TTL_SECONDS:,}s to {NEW_TTL_SECONDS:,}s baseline")
    print(f"Pattern: {STM_KEY_PATTERN}")
    print()
    
    # Run the migration
    asyncio.run(migrate_stm_ttls())
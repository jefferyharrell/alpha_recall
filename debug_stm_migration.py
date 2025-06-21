#!/usr/bin/env python3
"""
DEBUG VERSION: STM embedding migration with extensive logging.

This version adds detailed debugging to understand why the safety check
works but the main migration fails on identical data.
"""

import asyncio
import json
import struct
import sys
from datetime import datetime
from typing import List, Optional

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


def debug_hash_data(key: str, data: dict, phase: str):
    """Debug helper to examine hash data structure."""
    print(f"\n  🔍 DEBUG [{phase}] Key: {key}")
    print(f"     Total fields: {len(data)}")
    print(f"     Field names: {list(data.keys())}")
    print(f"     Field types: {[type(k) for k in data.keys()]}")
    print(f"     Has b'content': {b'content' in data}")
    print(f"     Has 'content': {'content' in data}")
    
    # Try to get content both ways
    content_bytes = data.get(b'content')
    content_str = data.get('content')
    
    print(f"     data.get(b'content'): {type(content_bytes)} = {repr(content_bytes)[:50]}...")
    print(f"     data.get('content'): {type(content_str)} = {repr(content_str)[:50]}...")
    
    # Try our extraction logic
    content = content_bytes or content_str
    print(f"     Final content: {type(content)} = {repr(content)[:50] if content else 'None'}...")
    
    return content


async def test_content_extraction(client, keys: List[bytes], phase: str) -> int:
    """Test content extraction on a set of keys."""
    print(f"\n🧪 TESTING CONTENT EXTRACTION - {phase}")
    readable_count = 0
    
    for i, key in enumerate(keys, 1):
        try:
            print(f"\n  [{i}/{len(keys)}] Testing {key.decode()}")
            
            # Get hash data
            data = await client.hgetall(key)
            
            # Debug the data structure
            content = debug_hash_data(key.decode(), data, phase)
            
            if content:
                # Decode if bytes, otherwise use as string
                content_str = content.decode('utf-8') if isinstance(content, bytes) else content
                if content_str.strip():  # Make sure it's not just whitespace
                    readable_count += 1
                    print(f"     ✅ SUCCESS: Content readable ({len(content_str)} chars)")
                else:
                    print(f"     ❌ FAIL: Content is empty/whitespace")
            else:
                print(f"     ❌ FAIL: No content found")
                
        except Exception as e:
            print(f"     💥 EXCEPTION: {e}")
    
    print(f"\n📊 {phase} RESULTS: {readable_count}/{len(keys)} readable")
    return readable_count


async def debug_stm_migration():
    """Main debug function with extensive logging."""
    print("🔍 DEBUG STM Migration - Detailed Analysis")
    print("=" * 60)
    
    print(f"\n📡 Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    # Connect to Redis
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=False,  # Keep binary support
    )
    
    try:
        await client.ping()
        print("✅ Connected to Redis")
        
        # Get Redis info
        info = await client.info()
        print(f"📈 Redis version: {info.get('redis_version', 'unknown')}")
        print(f"🔢 Database size: {info.get('db0', {}).get('keys', 0)} keys")
        
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        return
    
    # Test embedder service
    print(f"\n🧠 Testing new embedder service...")
    test_embedding = await get_new_embedding("Test connection")
    if test_embedding is None:
        print("❌ Failed to connect to embedder service")
        return
    print(f"✅ Embedder service working, returned {len(test_embedding)}D vector")
    
    # Get all STM keys
    print(f"\n🔍 Scanning for STM entries with prefix: {KEY_PREFIX}")
    keys = []
    cursor = 0
    while True:
        cursor, batch = await client.scan(
            cursor, match=f"{KEY_PREFIX}*", count=100
        )
        keys.extend(batch)
        if cursor == 0:
            break
    
    print(f"📦 Found {len(keys)} STM entries")
    
    if not keys:
        print("❌ No STM entries found. Exiting.")
        return
    
    # Show key sample
    print(f"🎯 First 5 keys: {[k.decode() for k in keys[:5]]}")
    
    # PHASE 1: Safety check (original logic)
    test_keys = keys[:5] if len(keys) >= 5 else keys
    safety_readable = await test_content_extraction(client, test_keys, "SAFETY CHECK")
    
    if safety_readable == 0:
        print(f"\n🛑 SAFETY CHECK FAILED: Cannot read any content!")
        await client.close()
        return
    
    # PHASE 2: Test the exact same keys but in main migration order
    print(f"\n🔄 Re-testing same keys in migration order...")
    migration_readable = await test_content_extraction(client, test_keys, "MIGRATION ORDER")
    
    # PHASE 3: Test a larger batch
    larger_batch = keys[:20] if len(keys) >= 20 else keys
    print(f"\n📊 Testing larger batch ({len(larger_batch)} keys)...")
    batch_readable = await test_content_extraction(client, larger_batch, "LARGE BATCH")
    
    # PHASE 4: Test connection state after operations
    print(f"\n🔄 Re-testing first key after many operations...")
    first_key_retest = await test_content_extraction(client, [keys[0]], "CONNECTION STATE")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📋 FINAL ANALYSIS")
    print(f"   Safety check readable: {safety_readable}/{len(test_keys)}")
    print(f"   Migration order readable: {migration_readable}/{len(test_keys)}")
    print(f"   Large batch readable: {batch_readable}/{len(larger_batch)}")
    print(f"   Connection state readable: {first_key_retest}/1")
    
    if safety_readable != migration_readable:
        print(f"\n🚨 INCONSISTENCY DETECTED!")
        print(f"   Same keys, different results between phases")
        print(f"   This suggests a Redis connection or state issue")
    elif batch_readable < len(larger_batch) * 0.8:  # 80% threshold
        print(f"\n⚠️  BATCH SIZE ISSUE!")
        print(f"   Large batches have lower success rate")
        print(f"   This suggests batch processing problems")
    elif safety_readable == len(test_keys):
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"   Data is readable, bug must be elsewhere")
    else:
        print(f"\n❓ PARTIAL SUCCESS")
        print(f"   Some data readable, some not")
    
    # Connection cleanup
    await client.close()
    print(f"\n📡 Redis connection closed")


if __name__ == "__main__":
    asyncio.run(debug_stm_migration())
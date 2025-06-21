#!/usr/bin/env python3
"""
Test script to verify the new 768D embedder integration.

This script tests:
1. New semantic embedding endpoint (768D)
2. Emotional embedding endpoint (1024D)
3. Batch embedding functionality
4. Redis integration with new endpoints
5. Qdrant integration with new endpoints
"""

import asyncio
import sys
from typing import Optional

import httpx
import numpy as np

# Configuration
NEW_SEMANTIC_URL = "http://localhost:6004/api/v1/embeddings/semantic"
NEW_EMOTIONAL_URL = "http://localhost:6004/api/v1/embeddings/emotion"
NEW_BATCH_URL = "http://localhost:6004/api/v1/embeddings/batch"


async def test_semantic_embedding():
    """Test the new semantic embedding endpoint."""
    print("Testing semantic embedding endpoint...")
    
    try:
        payload = {"texts": ["Alpha is testing the new embedding system"]}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                NEW_SEMANTIC_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
            
            response.raise_for_status()
            data = response.json()
            
            embeddings = data.get("embeddings")
            if not embeddings or not isinstance(embeddings, list):
                print(f"  ✗ No embeddings in response: {data}")
                return False
                
            embedding_array = np.array(embeddings[0])
            print(f"  ✓ Received {len(embedding_array)}D semantic embedding")
            print(f"  ✓ Shape: {embedding_array.shape}, dtype: {embedding_array.dtype}")
            return True
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def test_emotional_embedding():
    """Test the emotional embedding endpoint."""
    print("\nTesting emotional embedding endpoint...")
    
    try:
        payload = {"texts": ["I'm excited about this new upgrade!"]}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                NEW_EMOTIONAL_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
            
            response.raise_for_status()
            data = response.json()
            
            embeddings = data.get("embeddings")
            if not embeddings or not isinstance(embeddings, list):
                print(f"  ✗ No embeddings in response: {data}")
                return False
                
            embedding_array = np.array(embeddings[0])
            print(f"  ✓ Received {len(embedding_array)}D emotional embedding")
            print(f"  ✓ Shape: {embedding_array.shape}, dtype: {embedding_array.dtype}")
            return True
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def test_batch_embedding():
    """Test the batch embedding endpoint."""
    print("\nTesting batch embedding endpoint...")
    
    try:
        payload = {
            "texts": [
                "First test sentence",
                "Second test sentence", 
                "Third test sentence"
            ]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                NEW_BATCH_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15.0,
            )
            
            response.raise_for_status()
            data = response.json()
            
            semantic_embeddings = data.get("semantic_embeddings")
            emotional_embeddings = data.get("emotional_embeddings")
            
            if not semantic_embeddings or not emotional_embeddings:
                print(f"  ✗ Missing embeddings in response: {data}")
                return False
            
            print(f"  ✓ Received {len(semantic_embeddings)} semantic embeddings")
            print(f"  ✓ Received {len(emotional_embeddings)} emotional embeddings")
            print(f"  ✓ Semantic: {len(semantic_embeddings[0])}D each")
            print(f"  ✓ Emotional: {len(emotional_embeddings[0])}D each")
            return True
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def test_alpha_recall_integration():
    """Test alpha-recall integration with new embedder."""
    print("\nTesting alpha-recall integration...")
    
    try:
        # Import alpha-recall modules
        sys.path.append('/Users/jefferyharrell/Projects/Alpha/Alpha-Recall/alpha_recall/src')
        from alpha_recall.db.factory import create_shortterm_memory, create_vector_store
        
        # Test Redis STM integration
        print("  Testing Redis STM integration...")
        stm = await create_shortterm_memory()
        if stm and await stm.is_connected():
            print("    ✓ Redis connection successful")
            
            # Test embedding generation
            test_embedding = await stm._embed_text("Test memory for new embedder")
            if test_embedding is not None:
                print(f"    ✓ Generated {len(test_embedding)}D embedding via Redis")
            else:
                print("    ✗ Failed to generate embedding via Redis")
                return False
                
            await stm.close()
        else:
            print("    ✗ Failed to connect to Redis")
            return False
        
        # Test Qdrant integration
        print("  Testing Qdrant vector store integration...")
        vector_store = create_vector_store()
        test_embedding = await vector_store.embed_text("Test observation for new embedder")
        if test_embedding is not None:
            print(f"    ✓ Generated {len(test_embedding)}D embedding via Qdrant")
        else:
            print("    ✗ Failed to generate embedding via Qdrant")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Integration error: {e}")
        return False


async def main():
    """Run all tests."""
    print("Alpha-Recall New Embedder Integration Test")
    print("=" * 50)
    
    tests = [
        test_semantic_embedding,
        test_emotional_embedding,
        test_batch_embedding,
        test_alpha_recall_integration,
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Semantic Embedding",
        "Emotional Embedding", 
        "Batch Embedding",
        "Alpha-Recall Integration"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for migration.")
    else:
        print("❌ Some tests failed. Fix issues before migrating.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
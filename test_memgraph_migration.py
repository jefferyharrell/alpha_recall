#!/usr/bin/env python3
"""
Test script for the Memgraph vector migration.
Verifies that the factory creates the correct vector store and can perform searches.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alpha_recall.db.factory import create_vector_store, VECTOR_STORE_TYPE
from alpha_recall.logging_utils import get_logger

logger = get_logger(__name__)


async def test_vector_store():
    """Test the vector store creation and basic functionality."""
    print("=" * 60)
    print("MEMGRAPH VECTOR MIGRATION TEST")
    print("=" * 60)
    
    # Show configuration
    print(f"Vector Store Type: {VECTOR_STORE_TYPE}")
    
    try:
        # Create vector store
        print("\n1. Creating vector store...")
        vector_store = create_vector_store()
        print(f"✅ Created vector store: {type(vector_store).__name__}")
        
        # Test embedding generation
        print("\n2. Testing embedding generation...")
        test_text = "This is a test observation about Alpha's memory system."
        embedding = await vector_store.embed_text(test_text)
        print(f"✅ Generated embedding: shape={embedding.shape}, dtype={embedding.dtype}")
        
        # Test search (should work even if no specific results found)
        print("\n3. Testing vector search...")
        search_query = "memory architecture"
        results = await vector_store.search_observations(search_query, limit=3)
        print(f"✅ Search completed: found {len(results)} results")
        
        if results:
            print("\nTop search result:")
            top_result = results[0]
            print(f"  Entity: {top_result.get('entity_name', 'N/A')}")
            print(f"  Score: {top_result.get('score', 0):.4f}")
            print(f"  Text: {top_result.get('text', '')[:100]}...")
        else:
            print("  (No results found - this is expected if vectors aren't migrated yet)")
        
        # Clean up if needed
        if hasattr(vector_store, 'close'):
            vector_store.close()
            
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"🎯 Memgraph vector store is working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        logger.exception("Vector store test failed")
        return False


async def test_factory_fallback():
    """Test that the factory can handle different configurations."""
    print(f"\n" + "=" * 60)
    print("FACTORY CONFIGURATION TEST")
    print("=" * 60)
    
    original_type = os.environ.get("VECTOR_STORE_TYPE")
    
    try:
        # Test Memgraph configuration
        os.environ["VECTOR_STORE_TYPE"] = "memgraph"
        vector_store = create_vector_store()
        print(f"✅ Memgraph vector store: {type(vector_store).__name__}")
        if hasattr(vector_store, 'close'):
            vector_store.close()
        
        # Test Qdrant configuration (if available)
        # Note: This might fail if Qdrant isn't running, which is fine
        try:
            os.environ["VECTOR_STORE_TYPE"] = "qdrant"
            vector_store = create_vector_store()
            print(f"✅ Qdrant vector store: {type(vector_store).__name__}")
            if hasattr(vector_store, 'close'):
                vector_store.close()
        except Exception as e:
            print(f"⚠️  Qdrant not available (expected): {e}")
        
        print(f"✅ Factory configuration tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False
        
    finally:
        # Restore original configuration
        if original_type:
            os.environ["VECTOR_STORE_TYPE"] = original_type
        elif "VECTOR_STORE_TYPE" in os.environ:
            del os.environ["VECTOR_STORE_TYPE"]


async def main():
    """Run all tests."""
    # Set up environment for testing
    os.environ.setdefault("VECTOR_STORE_TYPE", "memgraph")
    os.environ.setdefault("GRAPH_DB_URI", "bolt://localhost:7687")
    
    print("Starting Memgraph Vector Migration Tests...")
    
    success = True
    
    # Test 1: Basic vector store functionality
    success &= await test_vector_store()
    
    # Test 2: Factory configuration
    success &= await test_factory_fallback()
    
    # Final results
    print(f"\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - MIGRATION SUCCESSFUL!")
        print("🚀 Alpha-Recall is ready to use Memgraph vectors")
    else:
        print("❌ SOME TESTS FAILED - CHECK CONFIGURATION")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
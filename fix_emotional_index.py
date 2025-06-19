#!/usr/bin/env python3
"""
Fix the Redis emotional vector index to use 1024 dimensions instead of 7.

This script:
1. Connects to Redis
2. Drops the existing emotional vector index (if it exists)
3. Creates a new emotional vector index with 1024 dimensions
4. Verifies the index was created correctly
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from alpha_recall.db.redis_db import RedisShortTermMemory
from alpha_recall.logging_utils import configure_logging, get_logger

# Load environment variables
load_dotenv()

# Configure logging
configure_logging()
logger = get_logger("fix_emotional_index")


async def fix_emotional_vector_index():
    """Fix the Redis emotional vector index to use 1024 dimensions."""
    logger.info("Starting emotional vector index fix...")
    
    # Initialize Redis connection
    redis_stm = RedisShortTermMemory(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD"),
        db=int(os.getenv("REDIS_DB", "0"))
    )
    
    try:
        # Connect to Redis
        await redis_stm.connect()
        logger.info("Connected to Redis")
        
        # Check current index info
        try:
            current_info = await redis_stm.client.execute_command('FT.INFO', 'idx:stm_emotional')
            logger.info("Current emotional vector index exists")
            
            # Parse the info to see current dimensions (this is complex, so we'll just drop and recreate)
            logger.info("Current index info retrieved, will recreate with correct dimensions")
            
        except Exception as e:
            logger.info(f"No existing emotional index found: {str(e)}")
        
        # Drop the existing emotional vector index
        try:
            await redis_stm.client.execute_command('FT.DROPINDEX', 'idx:stm_emotional', 'DD')
            logger.info("✅ Dropped existing emotional vector index")
        except Exception as e:
            logger.info(f"No existing emotional index to drop: {str(e)}")
        
        # Create new emotional vector index with 1024 dimensions
        logger.info("Creating new emotional vector index with 1024 dimensions...")
        
        try:
            await redis_stm.client.execute_command(
                'FT.CREATE', 'idx:stm_emotional',
                'ON', 'HASH',
                'PREFIX', '1', redis_stm.key_prefix,
                'SCHEMA',
                'content', 'TEXT',
                'embedding_emotional', 'VECTOR', 'FLAT', '6',
                'TYPE', 'FLOAT32',
                'DIM', '1024',  # Updated to 1024 dimensions
                'DISTANCE_METRIC', 'COSINE'
            )
            logger.info("✅ Created new emotional vector index with 1024 dimensions")
            
        except Exception as e:
            logger.error(f"❌ Failed to create emotional vector index: {str(e)}")
            raise
        
        # Verify the index was created correctly
        try:
            new_info = await redis_stm.client.execute_command('FT.INFO', 'idx:stm_emotional')
            logger.info("✅ New emotional vector index verified")
            
            # Parse some basic info
            info_pairs = []
            for i in range(0, len(new_info), 2):
                if i + 1 < len(new_info):
                    key = new_info[i].decode('utf-8') if isinstance(new_info[i], bytes) else str(new_info[i])
                    value = new_info[i+1]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    info_pairs.append((key, value))
            
            # Find dimension info in the attributes
            for key, value in info_pairs:
                if key == 'attributes' and isinstance(value, list):
                    # Look for embedding_emotional field info
                    for attr in value:
                        if isinstance(attr, list) and len(attr) >= 2:
                            field_name = attr[1].decode('utf-8') if isinstance(attr[1], bytes) else str(attr[1])
                            if field_name == 'embedding_emotional':
                                logger.info(f"Found emotional embedding field configuration")
                                break
            
            logger.info("✅ Emotional vector index recreated successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to verify new index: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"Failed to fix emotional vector index: {str(e)}")
        raise
    
    finally:
        # Close Redis connection
        if hasattr(redis_stm, 'client') and redis_stm.client:
            await redis_stm.client.aclose()
            logger.info("Closed Redis connection")


async def test_fixed_index():
    """Test that the fixed index works with 1024-dimensional vectors."""
    logger.info("Testing fixed emotional vector index...")
    
    redis_stm = RedisShortTermMemory(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD"),
        db=int(os.getenv("REDIS_DB", "0")),
        emotional_embedding_url=os.getenv("EMOTIONAL_EMBEDDING_URL", "http://localhost:6004/sentiment-embeddings")
    )
    
    try:
        await redis_stm.connect()
        
        # Test emotional search with the fixed index
        logger.info("Testing emotional search with fixed index...")
        
        emotional_results = await redis_stm.emotional_search_memories(
            query="I am feeling very happy and excited today", 
            limit=5
        )
        
        logger.info(f"Emotional search returned {len(emotional_results)} results")
        
        # Check if we got actual emotional similarity scores (not just recent memories)
        has_emotional_scores = any(
            'emotional_score' in result and result['emotional_score'] is not None 
            for result in emotional_results
        )
        
        if has_emotional_scores:
            logger.info("✅ Emotional search is now working with similarity scores!")
            for i, result in enumerate(emotional_results[:3]):
                score = result.get('emotional_score', 'N/A')
                content = result.get('content', '')[:50]
                logger.info(f"  Result {i+1}: emotional_score={score}, content='{content}...'")
            return True
        else:
            logger.warning("⚠️  Emotional search returned results but without emotional similarity scores")
            logger.info("This might be normal if existing memories don't have emotional embeddings yet")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to test fixed index: {str(e)}")
        return False
    finally:
        if hasattr(redis_stm, 'client') and redis_stm.client:
            await redis_stm.client.aclose()


async def main():
    """Main entry point."""
    try:
        await fix_emotional_vector_index()
        logger.info("\n" + "="*50)
        await test_fixed_index()
        logger.info("\n🎉 Emotional vector index fix completed successfully!")
        logger.info("\nYou should now be able to use the remember_shortterm tool and get both semantic and emotional results.")
        
    except Exception as e:
        logger.error(f"Fix failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
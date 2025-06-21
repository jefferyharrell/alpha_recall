#!/usr/bin/env python3
"""
Migrate long-term memory observations from 384D to 768D semantic embeddings.

This script:
1. Connects to Qdrant and the new embedder service
2. Creates a new collection for 768D embeddings
3. Fetches all observations from the old collection
4. Re-embeds their content using the new 768D model
5. Stores them in the new collection
6. Optionally swaps the collections
"""

import asyncio
import sys
from datetime import datetime
from typing import List, Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = "http://localhost:6333"
OLD_COLLECTION = "alpha_recall_observations"
NEW_COLLECTION = "alpha_recall_observations_768d"
BATCH_SIZE = 50  # Process in batches to avoid memory issues

# New embedder service configuration
NEW_EMBEDDER_URL = "http://localhost:6004/api/v1/embeddings/semantic"
NEW_VECTOR_SIZE = 768

# Old configuration
OLD_VECTOR_SIZE = 384


async def get_new_embedding(text: str) -> Optional[np.ndarray]:
    """Generate 768D embedding using the new embedder service."""
    try:
        payload = {"texts": [text]}
        
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                NEW_EMBEDDER_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from semantic endpoint response
            embeddings = data.get("embeddings")
            if not embeddings or not isinstance(embeddings, list):
                print(f"Error: No embeddings in response: {data}")
                return None
                
            return np.array(embeddings[0], dtype=np.float32)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


async def get_batch_embeddings(texts: List[str]) -> Optional[List[np.ndarray]]:
    """Generate multiple 768D embeddings in a batch."""
    try:
        payload = {"texts": texts}
        
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                NEW_EMBEDDER_URL.replace('/semantic', '/batch'),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60.0,
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract semantic embeddings from batch response
            semantic_data = data.get("semantic", {})
            embeddings = semantic_data.get("embeddings", [])
            if not embeddings:
                print(f"Error: No semantic embeddings in batch response")
                return None
                
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
    except Exception as e:
        # Fall back to individual embeddings if batch fails
        print(f"Batch embedding failed, falling back to individual: {e}")
        embeddings = []
        for text in texts:
            emb = await get_new_embedding(text)
            if emb is None:
                return None
            embeddings.append(emb)
        return embeddings


async def migrate_ltm_embeddings():
    """Main migration function."""
    print("Starting LTM embedding migration from 384D to 768D...")
    print(f"Connecting to Qdrant at {QDRANT_URL}")
    
    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL)
    
    try:
        collections = client.get_collections().collections
        print(f"✓ Connected to Qdrant, found {len(collections)} collections")
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        return
    
    # Check if old collection exists
    if not client.collection_exists(OLD_COLLECTION):
        print(f"✗ Collection '{OLD_COLLECTION}' not found")
        return
    
    # Get collection info
    old_info = client.get_collection(OLD_COLLECTION)
    total_points = old_info.points_count
    print(f"✓ Found collection '{OLD_COLLECTION}' with {total_points} points")
    
    # Safety limit to prevent infinite loops
    max_batches = (total_points // BATCH_SIZE) + 2  # Allow 2 extra batches for safety
    
    # Test embedder service
    print("\nTesting new embedder service...")
    test_embedding = await get_new_embedding("Test connection")
    if test_embedding is None:
        print("✗ Failed to connect to embedder service")
        return
    print(f"✓ Embedder service working, returned {len(test_embedding)}D vector")
    
    # Create new collection
    print(f"\nCreating new collection '{NEW_COLLECTION}'...")
    try:
        if client.collection_exists(NEW_COLLECTION):
            print(f"Collection '{NEW_COLLECTION}' already exists. Delete it? (y/n): ", end="")
            if input().lower() == 'y':
                client.delete_collection(NEW_COLLECTION)
                print("✓ Deleted existing collection")
            else:
                print("Exiting without changes")
                return
        
        client.create_collection(
            collection_name=NEW_COLLECTION,
            vectors_config=models.VectorParams(
                size=NEW_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"✓ Created new collection with {NEW_VECTOR_SIZE}D vectors")
    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        return
    
    # Migrate observations
    print("\nMigrating observations...")
    total_migrated = 0
    error_count = 0
    next_page_offset = None
    batch_num = 0
    seen_point_ids = set()  # Track processed points to detect loops
    
    while batch_num < max_batches:
        # Fetch batch of points using proper scroll pagination
        scroll_result = client.scroll(
            collection_name=OLD_COLLECTION,
            limit=BATCH_SIZE,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False,  # We don't need the old vectors
        )
        points, next_page_offset = scroll_result
        
        if not points:
            break
        
        batch_num += 1
        
        # Check for duplicate points (indicates pagination loop)
        current_ids = {point.id for point in points}
        if current_ids & seen_point_ids:  # Intersection means duplicates
            print(f"\n⚠️  Detected duplicate points in batch {batch_num} - stopping to prevent infinite loop")
            break
        seen_point_ids.update(current_ids)
        
        start_idx = total_migrated + 1
        end_idx = total_migrated + len(points)
        print(f"\nProcessing batch {batch_num} (points {start_idx}-{end_idx})...")
        
        # Extract texts for batch embedding
        texts = []
        valid_points = []
        
        for point in points:
            text = point.payload.get("text")
            if text:
                texts.append(text)
                valid_points.append(point)
            else:
                print(f"  Warning: Point {point.id} has no text, skipping")
        
        if not texts:
            continue
        
        # Generate new embeddings
        try:
            new_embeddings = await get_batch_embeddings(texts)
            if new_embeddings is None:
                print(f"  ✗ Failed to generate embeddings for batch")
                error_count += len(texts)
                continue
        except Exception as e:
            print(f"  ✗ Error generating embeddings: {e}")
            error_count += len(texts)
            continue
        
        # Create new points with updated embeddings
        new_points = []
        for point, embedding in zip(valid_points, new_embeddings):
            new_points.append(
                models.PointStruct(
                    id=point.id,
                    vector=embedding.tolist(),
                    payload=point.payload,
                )
            )
        
        # Upload to new collection
        try:
            client.upsert(
                collection_name=NEW_COLLECTION,
                points=new_points,
            )
            total_migrated += len(new_points)
            print(f"  ✓ Migrated {len(new_points)} observations")
        except Exception as e:
            print(f"  ✗ Error uploading batch: {e}")
            error_count += len(new_points)
    
    print(f"\nMigration complete: {total_migrated} successful, {error_count} errors")
    
    # Verify the new collection
    new_info = client.get_collection(NEW_COLLECTION)
    print(f"\nNew collection '{NEW_COLLECTION}' has {new_info.points_count} points")
    
    # Option to swap collections
    if total_migrated > 0 and error_count == 0:
        print(f"\nSwap collections? This will:")
        print(f"  1. Rename '{OLD_COLLECTION}' to '{OLD_COLLECTION}_backup'")
        print(f"  2. Rename '{NEW_COLLECTION}' to '{OLD_COLLECTION}'")
        print(f"Continue? (y/n): ", end="")
        
        if input().lower() == 'y':
            try:
                # Create aliases for atomic swap
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{OLD_COLLECTION}_backup_{timestamp}"
                
                # Delete old collection (Qdrant doesn't support rename)
                print(f"Backing up old collection as '{backup_name}'...")
                client.update_collection_aliases(
                    change_aliases_operations=[
                        models.CreateAliasOperation(
                            create_alias=models.CreateAlias(
                                collection_name=OLD_COLLECTION,
                                alias_name=backup_name,
                            )
                        )
                    ]
                )
                
                # Point main collection name to new collection
                client.update_collection_aliases(
                    change_aliases_operations=[
                        models.CreateAliasOperation(
                            create_alias=models.CreateAlias(
                                collection_name=NEW_COLLECTION,
                                alias_name=OLD_COLLECTION,
                            )
                        )
                    ]
                )
                
                print("✓ Collections swapped successfully")
                print(f"  Old collection backed up with alias: {backup_name}")
                print(f"  New collection available as: {OLD_COLLECTION}")
            except Exception as e:
                print(f"✗ Failed to swap collections: {e}")
                print("  You can manually swap them later")
    
    print("\nMigration complete!")


if __name__ == "__main__":
    asyncio.run(migrate_ltm_embeddings())
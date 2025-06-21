"""
Narrative memory storage implementation for Alpha-Recall.

This module provides hybrid storage for narrative memories using:
- Redis: Story content, embeddings, and vector search
- Memgraph: Story nodes, metadata, and relationship tracking
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np
import redis.asyncio as redis

from alpha_recall.logging_utils import get_logger

logger = get_logger(__name__)

# Default configuration
DEFAULT_NARRATIVE_KEY_PREFIX = "narrative:"
DEFAULT_STORY_INDEX_NAME = "idx:narrative_stories"
DEFAULT_PARAGRAPH_INDEX_NAME = "idx:narrative_paragraphs"


class NarrativeMemory:
    """
    Hybrid narrative memory storage combining Redis (content + vectors) and Memgraph (metadata + relationships).
    
    Story Storage Schema (Redis Hash):
    {
        "story_id": "unique_story_identifier",
        "title": "Story Title",
        "created_at": "2025-06-21T21:49:39.312479+00:00",
        "participants": ["Alpha", "Jeffery"],
        "tags": ["debugging", "breakthrough"],
        "paragraphs": "[{\"text\": \"...\", \"order\": 0}, ...]",
        "full_semantic_vector": "<768D binary blob>",
        "full_emotional_vector": "<1024D binary blob>",
        "para_0_semantic": "<768D binary blob>",
        "para_0_emotional": "<1024D binary blob>",
        "para_1_semantic": "<768D binary blob>",
        "para_1_emotional": "<1024D binary blob>"
    }
    
    Story Node Schema (Memgraph):
    (:Story {
        story_id: "unique_story_identifier",
        title: "Story Title", 
        created_at: "2025-06-21T21:49:39.312479+00:00",
        outcome: "breakthrough|resolution|ongoing",
        paragraph_count: 3
    })
    
    Relationships:
    (:Story)-[:INVOLVES]->(:Entity {name: "Jeffery"})
    (:Story)-[:ABOUT]->(:Entity {name: "debugging"})
    (:Story)-[:REFERENCES]->(:Story {story_id: "previous_story"})
    (:Story)-[:LEADS_TO]->(:Story {story_id: "next_story"})
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        graph_db,
        embedding_server_url: str,
        emotional_embedding_url: str,
        key_prefix: str = DEFAULT_NARRATIVE_KEY_PREFIX,
    ):
        """
        Initialize narrative memory storage.

        Args:
            redis_client: Connected Redis client
            graph_db: Connected graph database (Memgraph/Neo4j)
            embedding_server_url: URL for semantic embeddings (768D)
            emotional_embedding_url: URL for emotional embeddings (1024D)
            key_prefix: Redis key prefix for narrative stories
        """
        self.redis = redis_client
        self.graph_db = graph_db
        self.embedding_server_url = embedding_server_url
        self.emotional_embedding_url = emotional_embedding_url
        self.key_prefix = key_prefix
        self._story_index_created = False
        self._paragraph_index_created = False

    async def store_story(
        self,
        title: str,
        paragraphs: List[str],
        participants: List[str],
        tags: Optional[List[str]] = None,
        outcome: str = "ongoing",
        references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Store a narrative story with dual-granularity embeddings and graph metadata.

        Args:
            title: Story title
            paragraphs: List of paragraph texts
            participants: List of participant names (e.g., ["Alpha", "Jeffery"])
            tags: Optional list of tags/topics
            outcome: Story outcome ("breakthrough", "resolution", "ongoing")
            references: Optional list of story_ids this story references

        Returns:
            Dictionary with story information and storage status
        """
        try:
            # Generate unique story ID
            story_id = f"story_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:8]}"
            
            # Create timestamp
            created_at = datetime.now(timezone.utc).isoformat()
            
            # Prepare paragraph objects with order
            paragraph_objects = [
                {"text": text, "order": i} for i, text in enumerate(paragraphs)
            ]
            
            # Generate embeddings for full story and individual paragraphs
            full_text = f"{title}\n\n" + "\n\n".join(paragraphs)
            
            # Get embeddings
            story_semantic = await self._get_semantic_embedding(full_text)
            story_emotional = await self._get_emotional_embedding(full_text)
            
            paragraph_embeddings = []
            for i, paragraph in enumerate(paragraphs):
                para_semantic = await self._get_semantic_embedding(paragraph)
                para_emotional = await self._get_emotional_embedding(paragraph)
                paragraph_embeddings.append({
                    "index": i,
                    "semantic": para_semantic,
                    "emotional": para_emotional
                })
            
            # Prepare Redis hash data
            story_data = {
                "story_id": story_id,
                "title": title,
                "created_at": created_at,
                "participants": json.dumps(participants),
                "tags": json.dumps(tags or []),
                "outcome": outcome,
                "paragraphs": json.dumps(paragraph_objects),
                "full_semantic_vector": story_semantic.tobytes(),
                "full_emotional_vector": story_emotional.tobytes(),
            }
            
            # Add paragraph-level embeddings
            for embed in paragraph_embeddings:
                i = embed["index"]
                story_data[f"para_{i}_semantic"] = embed["semantic"].tobytes()
                story_data[f"para_{i}_emotional"] = embed["emotional"].tobytes()
            
            # Store in Redis
            redis_key = f"{self.key_prefix}{story_id}"
            await self.redis.hset(redis_key, mapping=story_data)
            
            # Create vector indices if needed
            await self._ensure_vector_indices()
            
            # Create story node in graph database
            await self._create_story_node(
                story_id=story_id,
                title=title,
                created_at=created_at,
                outcome=outcome,
                paragraph_count=len(paragraphs),
                participants=participants,
                tags=tags or [],
                references=references or [],
            )
            
            logger.info(f"Successfully stored narrative story: {story_id}")
            
            return {
                "success": True,
                "story_id": story_id,
                "title": title,
                "created_at": created_at,
                "paragraph_count": len(paragraphs),
                "redis_key": redis_key,
            }
            
        except Exception as e:
            logger.error(f"Failed to store narrative story: {str(e)}")
            return {
                "success": False,
                "error": str(e),
            }

    async def search_stories(
        self,
        query: str,
        search_type: str = "semantic",  # "semantic", "emotional", "both"
        granularity: str = "story",     # "story", "paragraph", "both"  
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search narrative stories using vector similarity.

        Args:
            query: Search query text
            search_type: Type of search ("semantic", "emotional", "both")
            granularity: Search granularity ("story", "paragraph", "both")
            limit: Maximum number of results

        Returns:
            List of matching stories/paragraphs with similarity scores
        """
        try:
            results = []
            
            if search_type in ["semantic", "both"]:
                if granularity in ["story", "both"]:
                    # Story-level semantic search
                    story_results = await self._vector_search(
                        query=query,
                        vector_field="full_semantic_vector",
                        index_name=DEFAULT_STORY_INDEX_NAME,
                        limit=limit,
                        search_type="semantic"
                    )
                    results.extend(story_results)
                
                if granularity in ["paragraph", "both"]:
                    # Paragraph-level semantic search
                    para_results = await self._vector_search(
                        query=query,
                        vector_field="para_*_semantic",
                        index_name=DEFAULT_PARAGRAPH_INDEX_NAME,
                        limit=limit,
                        search_type="semantic",
                        granularity="paragraph"
                    )
                    results.extend(para_results)
            
            if search_type in ["emotional", "both"]:
                if granularity in ["story", "both"]:
                    # Story-level emotional search
                    story_results = await self._vector_search(
                        query=query,
                        vector_field="full_emotional_vector",
                        index_name=DEFAULT_STORY_INDEX_NAME,
                        limit=limit,
                        search_type="emotional"
                    )
                    results.extend(story_results)
                
                if granularity in ["paragraph", "both"]:
                    # Paragraph-level emotional search
                    para_results = await self._vector_search(
                        query=query,
                        vector_field="para_*_emotional",
                        index_name=DEFAULT_PARAGRAPH_INDEX_NAME,
                        limit=limit,
                        search_type="emotional",
                        granularity="paragraph"
                    )
                    results.extend(para_results)
            
            # Sort by score and deduplicate
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search narrative stories: {str(e)}")
            return []

    async def get_story(self, story_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete story by ID.

        Args:
            story_id: Unique story identifier

        Returns:
            Complete story data or None if not found
        """
        try:
            redis_key = f"{self.key_prefix}{story_id}"
            story_data = await self.redis.hgetall(redis_key)
            
            if not story_data:
                return None
            
            # Decode Redis hash data
            result = {}
            for key, value in story_data.items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if isinstance(value, bytes):
                    if key.endswith('_vector'):
                        # Skip vectors for readability
                        continue
                    else:
                        value = value.decode('utf-8')
                
                # Parse JSON fields
                if key in ["participants", "tags", "paragraphs"]:
                    result[key] = json.loads(value)
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve story {story_id}: {str(e)}")
            return None

    async def _get_semantic_embedding(self, text: str) -> np.ndarray:
        """Get 768D semantic embedding for text."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.embedding_server_url,
                json={"texts": [text]},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            # Extract the first (and only) embedding from the response
            return np.array(data["embeddings"][0], dtype=np.float32)

    async def _get_emotional_embedding(self, text: str) -> np.ndarray:
        """Get 1024D emotional embedding for text."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.emotional_embedding_url,
                json={"texts": [text]},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            # Extract the first (and only) embedding from the response
            return np.array(data["embeddings"][0], dtype=np.float32)

    async def _create_story_node(
        self,
        story_id: str,
        title: str,
        created_at: str,
        outcome: str,
        paragraph_count: int,
        participants: List[str],
        tags: List[str],
        references: List[str],
    ) -> None:
        """Create story node and relationships in graph database."""
        try:
            # Create story node
            await self.graph_db.create_entity(name=story_id, entity_type="Story")
            
            # Add story-specific properties (would need to extend graph_db for this)
            # For now, we'll create relationships to represent the metadata
            
            # Create relationships to participants
            for participant in participants:
                await self.graph_db.create_relationship(
                    source_entity=story_id,
                    target_entity=participant,
                    relationship_type="INVOLVES"
                )
            
            # Create relationships to tags/topics
            for tag in tags:
                await self.graph_db.create_relationship(
                    source_entity=story_id,
                    target_entity=tag,
                    relationship_type="ABOUT"
                )
            
            # Create references to other stories
            for ref_story_id in references:
                await self.graph_db.create_relationship(
                    source_entity=story_id,
                    target_entity=ref_story_id,
                    relationship_type="REFERENCES"
                )
                
        except Exception as e:
            logger.error(f"Failed to create story node for {story_id}: {str(e)}")

    async def _ensure_vector_indices(self) -> None:
        """Create Redis vector search indices if they don't exist."""
        try:
            if not self._story_index_created:
                # Create story-level vector index
                await self._create_vector_index(
                    index_name=DEFAULT_STORY_INDEX_NAME,
                    key_prefix=self.key_prefix,
                    vector_fields=["full_semantic_vector", "full_emotional_vector"],
                    semantic_dim=768,
                    emotional_dim=1024
                )
                self._story_index_created = True
            
            # Note: Paragraph-level indexing would require a different approach
            # since each story has variable numbers of paragraphs
            
        except Exception as e:
            logger.error(f"Failed to create vector indices: {str(e)}")

    async def _create_vector_index(
        self,
        index_name: str,
        key_prefix: str,
        vector_fields: List[str],
        semantic_dim: int,
        emotional_dim: int
    ) -> None:
        """Create a Redis vector search index."""
        # This would use Redis FT.CREATE command
        # Simplified implementation - would need full Redis vector index creation
        pass

    async def _vector_search(
        self,
        query: str,
        vector_field: str,
        index_name: str,
        limit: int,
        search_type: str,
        granularity: str = "story"
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        # This would use Redis FT.SEARCH with vector similarity
        # Simplified implementation - would need full Redis vector search
        return []
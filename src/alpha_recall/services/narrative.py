"""
Narrative memory service for Alpha-Recall v2.0.

This service provides hybrid storage for narrative memories using:
- Redis: Story content, embeddings, and vector search
- Memgraph: Story nodes, metadata, and relationship tracking

Compatible with alpha-recall 0.1.0 database schema.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import redis.asyncio as redis

from ..config import AlphaRecallSettings
from ..logging import get_logger
from ..services.embedding import EmbeddingService
from ..utils.correlation import generate_correlation_id, set_correlation_id

logger = get_logger("services.narrative")

# Redis configuration compatible with 0.1.0
DEFAULT_NARRATIVE_KEY_PREFIX = "narrative:"
DEFAULT_STORY_INDEX_NAME = "idx:narrative_stories"
DEFAULT_PARAGRAPH_INDEX_NAME = "idx:narrative_paragraphs"


class NarrativeService:
    """
    Narrative memory service compatible with alpha-recall 0.1.0 schema.

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
    """

    def __init__(
        self, embedding_service: EmbeddingService, settings: AlphaRecallSettings
    ):
        """
        Initialize narrative memory service.

        Args:
            embedding_service: Connected embedding service
            settings: Alpha-Recall configuration settings
        """
        self.embedding_service = embedding_service
        self.settings = settings
        self.key_prefix = DEFAULT_NARRATIVE_KEY_PREFIX

        # Redis connection will be established on demand
        self._redis_client: redis.Redis | None = None
        self._story_index_created = False
        self._paragraph_index_created = False

    async def _get_redis_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._redis_client is None:
            # Parse Redis URI
            redis_url = self.settings.redis_uri
            self._redis_client = redis.from_url(redis_url, decode_responses=False)
        else:
            # Check if the client is still usable (not closed)
            try:
                await self._redis_client.ping()
            except Exception:
                # Client is closed or unusable, create a new one
                redis_url = self.settings.redis_uri
                self._redis_client = redis.from_url(redis_url, decode_responses=False)
        return self._redis_client

    async def store_story(
        self,
        title: str,
        paragraphs: list[str],
        participants: list[str],
        tags: list[str] | None = None,
        outcome: str = "ongoing",
        references: list[str] | None = None,
    ) -> dict[str, Any]:
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
        correlation_id = generate_correlation_id("narrative_store")
        set_correlation_id(correlation_id)

        logger.info(
            "Storing narrative story",
            title=title,
            paragraph_count=len(paragraphs),
            participants=participants,
            outcome=outcome,
            tags=tags or [],
            references=references or [],
            correlation_id=correlation_id,
        )

        try:
            # Generate unique story ID compatible with 0.1.0 format
            timestamp = int(datetime.now(UTC).timestamp())
            story_id = f"story_{timestamp}_{uuid.uuid4().hex[:8]}"

            # Create timestamp in ISO format
            created_at = datetime.now(UTC).isoformat()

            # Prepare paragraph objects with order
            paragraph_objects = [
                {"text": text, "order": i} for i, text in enumerate(paragraphs)
            ]

            # Generate embeddings for full story and individual paragraphs
            full_text = f"{title}\n\n" + "\n\n".join(paragraphs)

            logger.info(
                "Generating embeddings",
                story_id=story_id,
                full_text_length=len(full_text),
                paragraph_count=len(paragraphs),
                correlation_id=correlation_id,
            )

            # Get embeddings using our EmbeddingService
            story_semantic = np.array(
                self.embedding_service.encode_semantic(full_text), dtype=np.float32
            )
            story_emotional = np.array(
                self.embedding_service.encode_emotional(full_text), dtype=np.float32
            )

            paragraph_embeddings = []
            for i, paragraph in enumerate(paragraphs):
                para_semantic = np.array(
                    self.embedding_service.encode_semantic(paragraph), dtype=np.float32
                )
                para_emotional = np.array(
                    self.embedding_service.encode_emotional(paragraph), dtype=np.float32
                )
                paragraph_embeddings.append(
                    {"index": i, "semantic": para_semantic, "emotional": para_emotional}
                )

            # Prepare Redis hash data (compatible with 0.1.0 schema)
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
            redis_client = await self._get_redis_client()
            redis_key = f"{self.key_prefix}{story_id}"
            await redis_client.hset(redis_key, mapping=story_data)

            # Create vector indices if needed
            await self._ensure_vector_indices()

            # TODO: Create story node in Memgraph
            # This would require integrating with the graph database
            # For now, we'll log it as a placeholder
            logger.info(
                "Story node creation placeholder",
                story_id=story_id,
                title=title,
                participants=participants,
                tags=tags or [],
                outcome=outcome,
                correlation_id=correlation_id,
            )

            logger.info(
                "Successfully stored narrative story",
                story_id=story_id,
                redis_key=redis_key,
                embeddings_generated=len(paragraph_embeddings)
                + 2,  # paragraphs + full story (semantic + emotional)
                correlation_id=correlation_id,
            )

            return {
                "success": True,
                "story_id": story_id,
                "title": title,
                "created_at": created_at,
                "paragraph_count": len(paragraphs),
                "redis_key": redis_key,
                "embeddings_generated": len(paragraph_embeddings) * 2
                + 2,  # Each para has 2 embeddings + full story has 2
                "storage_location": "hybrid_redis_memgraph",
                "correlation_id": correlation_id,
            }

        except Exception as e:
            logger.error(
                "Failed to store narrative story",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "correlation_id": correlation_id,
            }

    async def search_stories(
        self,
        query: str,
        search_type: str = "semantic",  # "semantic", "emotional", "both"
        granularity: str = "story",  # "story", "paragraph", "both"
        limit: int = 5,
    ) -> list[dict[str, Any]]:
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
        correlation_id = generate_correlation_id("narrative_search")
        set_correlation_id(correlation_id)

        logger.info(
            "Searching narrative stories",
            query=query[:100],  # Truncate long queries for logging
            search_type=search_type,
            granularity=granularity,
            limit=limit,
            correlation_id=correlation_id,
        )

        try:
            # Ensure limit is an integer
            limit = int(limit)
            results = []

            if search_type in ["semantic", "both"]:
                if granularity in ["story", "both"]:
                    # Story-level semantic search
                    story_results = await self._vector_search(
                        query=query,
                        vector_field="full_semantic_vector",
                        index_name=DEFAULT_STORY_INDEX_NAME,
                        limit=limit,
                        search_type="semantic",
                        correlation_id=correlation_id,
                    )
                    results.extend(story_results)

                if granularity in ["paragraph", "both"]:
                    # Paragraph-level semantic search
                    # Note: This is complex with current schema, would need separate implementation
                    logger.warning(
                        "Paragraph-level search not fully implemented",
                        search_type="semantic",
                        correlation_id=correlation_id,
                    )

            if search_type in ["emotional", "both"]:
                if granularity in ["story", "both"]:
                    # Story-level emotional search
                    story_results = await self._vector_search(
                        query=query,
                        vector_field="full_emotional_vector",
                        index_name=DEFAULT_STORY_INDEX_NAME,
                        limit=limit,
                        search_type="emotional",
                        correlation_id=correlation_id,
                    )
                    results.extend(story_results)

                if granularity in ["paragraph", "both"]:
                    # Paragraph-level emotional search
                    logger.warning(
                        "Paragraph-level search not fully implemented",
                        search_type="emotional",
                        correlation_id=correlation_id,
                    )

            # Sort by score and deduplicate
            results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            unique_results = []
            seen_story_ids = set()

            for result in results:
                story_id = result.get("story_id")
                if story_id and story_id not in seen_story_ids:
                    unique_results.append(result)
                    seen_story_ids.add(story_id)

                if len(unique_results) >= limit:
                    break

            logger.info(
                "Narrative search completed",
                query=query[:50],
                results_count=len(unique_results),
                search_type=search_type,
                granularity=granularity,
                correlation_id=correlation_id,
            )

            return unique_results

        except Exception as e:
            logger.error(
                "Failed to search narrative stories",
                query=query[:50],
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            return []

    async def get_story(self, story_id: str) -> dict[str, Any] | None:
        """
        Retrieve a complete story by ID.

        Args:
            story_id: Unique story identifier

        Returns:
            Complete story data or None if not found
        """
        correlation_id = generate_correlation_id("narrative_get")
        set_correlation_id(correlation_id)

        logger.info(
            "Retrieving narrative story",
            story_id=story_id,
            correlation_id=correlation_id,
        )

        try:
            redis_client = await self._get_redis_client()
            redis_key = f"{self.key_prefix}{story_id}"
            story_data = await redis_client.hgetall(redis_key)

            if not story_data:
                logger.info(
                    "Story not found",
                    story_id=story_id,
                    redis_key=redis_key,
                    correlation_id=correlation_id,
                )
                return None

            # Decode Redis hash data (compatible with 0.1.0 format)
            result = {}
            for key, value in story_data.items():
                if isinstance(key, bytes):
                    key = key.decode("utf-8")

                # Skip vector fields entirely (they contain binary data)
                if key.endswith("_vector") or (
                    key.startswith("para_")
                    and ("_semantic" in key or "_emotional" in key)
                ):
                    continue

                if isinstance(value, bytes):
                    value = value.decode("utf-8")

                # Parse JSON fields
                if key in ["participants", "tags", "paragraphs"]:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse JSON field {key}",
                            correlation_id=correlation_id,
                        )
                        result[key] = []
                else:
                    result[key] = value

            logger.info(
                "Successfully retrieved narrative story",
                story_id=story_id,
                title=result.get("title", "Unknown"),
                paragraph_count=len(result.get("paragraphs", [])),
                correlation_id=correlation_id,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to retrieve story",
                story_id=story_id,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            return None

    async def list_stories(
        self,
        limit: int = 10,
        offset: int = 0,
        since: str | None = None,
        participants: list[str] | None = None,
        tags: list[str] | None = None,
        outcome: str | None = None,
    ) -> dict[str, Any]:
        """
        List narrative stories chronologically with optional filtering.

        Args:
            limit: Maximum number of stories to return
            offset: Number of stories to skip (for pagination)
            since: Time window filter (e.g., "2d", "1w", "1m")
            participants: Filter by participants (AND logic)
            tags: Filter by tags (AND logic)
            outcome: Filter by outcome type

        Returns:
            Dictionary with stories list and pagination info
        """
        correlation_id = generate_correlation_id("narrative_list")
        set_correlation_id(correlation_id)

        logger.info(
            "Listing narrative stories",
            limit=limit,
            offset=offset,
            since=since,
            participants=participants,
            tags=tags,
            outcome=outcome,
            correlation_id=correlation_id,
        )

        try:
            redis_client = await self._get_redis_client()

            # Get all narrative keys from Redis
            pattern = f"{self.key_prefix}*"
            keys = []

            # Use SCAN to get all keys matching the pattern
            cursor = 0
            while True:
                cursor, batch = await redis_client.scan(
                    cursor, match=pattern, count=100
                )
                keys.extend(batch)
                if cursor == 0:
                    break

            if not keys:
                return {
                    "stories": [],
                    "total_count": 0,
                    "returned_count": 0,
                    "offset": offset,
                    "limit": limit,
                    "has_more": False,
                }

            # Get metadata for all stories
            stories_metadata = []
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")

                try:
                    # Get only the metadata fields we need (not the full story or vectors)
                    story_data = await redis_client.hmget(
                        key,
                        "story_id",
                        "title",
                        "created_at",
                        "participants",
                        "tags",
                        "outcome",
                    )

                    if story_data and story_data[0]:  # Check if story_id exists
                        metadata = {}
                        field_names = [
                            "story_id",
                            "title",
                            "created_at",
                            "participants",
                            "tags",
                            "outcome",
                        ]

                        for i, field_name in enumerate(field_names):
                            value = story_data[i]
                            if value is not None:
                                if isinstance(value, bytes):
                                    value = value.decode("utf-8")

                                # Parse JSON fields
                                if field_name in ["participants", "tags"]:
                                    try:
                                        metadata[field_name] = json.loads(value)
                                    except json.JSONDecodeError:
                                        metadata[field_name] = []
                                else:
                                    metadata[field_name] = value
                            else:
                                metadata[field_name] = (
                                    []
                                    if field_name in ["participants", "tags"]
                                    else None
                                )

                        stories_metadata.append(metadata)

                except Exception as e:
                    logger.warning(
                        f"Failed to get metadata for {key}: {str(e)}",
                        correlation_id=correlation_id,
                    )
                    continue

            # Apply filters (same logic as 0.1.0)
            filtered_stories = self._apply_story_filters(
                stories_metadata, since, participants, tags, outcome, correlation_id
            )

            # Sort by created_at descending (most recent first)
            filtered_stories.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            # Apply pagination
            total_count = len(filtered_stories)
            paginated_stories = filtered_stories[offset : offset + limit]

            logger.info(
                "Successfully listed narrative stories",
                total_count=total_count,
                returned_count=len(paginated_stories),
                offset=offset,
                limit=limit,
                correlation_id=correlation_id,
            )

            return {
                "stories": paginated_stories,
                "total_count": total_count,
                "returned_count": len(paginated_stories),
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(paginated_stories) < total_count,
            }

        except Exception as e:
            logger.error(
                "Failed to list stories",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            return {
                "stories": [],
                "total_count": 0,
                "returned_count": 0,
                "offset": offset,
                "limit": limit,
                "has_more": False,
                "error": str(e),
            }

    def _apply_story_filters(
        self,
        stories: list[dict[str, Any]],
        since: str | None,
        participants: list[str] | None,
        tags: list[str] | None,
        outcome: str | None,
        correlation_id: str,
    ) -> list[dict[str, Any]]:
        """Apply filtering logic to story list (compatible with 0.1.0)."""
        filtered_stories = []

        for story in stories:
            # Time filter
            if since and story.get("created_at"):
                try:
                    from datetime import datetime, timedelta

                    story_time = datetime.fromisoformat(
                        story["created_at"].replace("Z", "+00:00")
                    )
                    current_time = datetime.now(UTC)

                    # Parse since parameter (e.g., "2d", "1w", "1m")
                    if since.endswith("d"):
                        delta = timedelta(days=int(since[:-1]))
                    elif since.endswith("w"):
                        delta = timedelta(weeks=int(since[:-1]))
                    elif since.endswith("m"):
                        delta = timedelta(
                            days=int(since[:-1]) * 30
                        )  # Approximate month
                    elif since.endswith("h"):
                        delta = timedelta(hours=int(since[:-1]))
                    else:
                        delta = timedelta(days=1)  # Default to 1 day

                    if story_time < current_time - delta:
                        continue
                except Exception as e:
                    logger.warning(
                        f"Failed to parse time filter: {str(e)}",
                        correlation_id=correlation_id,
                    )
                    continue

            # Participants filter (AND logic)
            if participants:
                story_participants = story.get("participants", [])
                if not all(p in story_participants for p in participants):
                    continue

            # Tags filter (AND logic)
            if tags:
                story_tags = story.get("tags", [])
                if not all(t in story_tags for t in tags):
                    continue

            # Outcome filter
            if outcome and story.get("outcome") != outcome:
                continue

            filtered_stories.append(story)

        return filtered_stories

    async def _ensure_vector_indices(self) -> None:
        """Create Redis vector search indices if they don't exist."""
        if not self._story_index_created:
            try:
                await self._create_vector_index(
                    index_name=DEFAULT_STORY_INDEX_NAME,
                    key_prefix=self.key_prefix,
                    semantic_dim=768,
                    emotional_dim=1024,
                )
                self._story_index_created = True
                logger.info(
                    "Story-level vector index ensured",
                    index_name=DEFAULT_STORY_INDEX_NAME,
                )
            except Exception as e:
                logger.error(f"Failed to create story vector index: {str(e)}")

    async def _create_vector_index(
        self,
        index_name: str,
        key_prefix: str,
        semantic_dim: int = 768,
        emotional_dim: int = 1024,
    ) -> None:
        """Create a Redis vector search index (compatible with 0.1.0 schema)."""
        try:
            redis_client = await self._get_redis_client()

            # Check if index already exists
            try:
                await redis_client.execute_command("FT.INFO", index_name)
                logger.info(f"Vector index {index_name} already exists")
                return
            except Exception:
                # Index doesn't exist, create it
                pass

            # Build FT.CREATE command for story-level vectors (compatible with 0.1.0)
            create_cmd = [
                "FT.CREATE",
                index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                key_prefix,
                "SCHEMA",
            ]

            # Add semantic vector field (768D)
            create_cmd.extend(
                [
                    "full_semantic_vector",
                    "VECTOR",
                    "FLAT",
                    "6",
                    "TYPE",
                    "FLOAT32",
                    "DIM",
                    str(semantic_dim),
                    "DISTANCE_METRIC",
                    "COSINE",
                ]
            )

            # Add emotional vector field (1024D)
            create_cmd.extend(
                [
                    "full_emotional_vector",
                    "VECTOR",
                    "FLAT",
                    "6",
                    "TYPE",
                    "FLOAT32",
                    "DIM",
                    str(emotional_dim),
                    "DISTANCE_METRIC",
                    "COSINE",
                ]
            )

            # Add metadata fields for filtering
            create_cmd.extend(
                [
                    "title",
                    "TEXT",
                    "participants",
                    "TAG",
                    "SEPARATOR",
                    ",",
                    "tags",
                    "TAG",
                    "SEPARATOR",
                    ",",
                    "outcome",
                    "TAG",
                ]
            )

            # Execute the command
            await redis_client.execute_command(*create_cmd)
            logger.info(f"Successfully created vector index: {index_name}")

        except Exception as e:
            logger.error(f"Failed to create vector index {index_name}: {str(e)}")
            raise

    async def _vector_search(
        self,
        query: str,
        vector_field: str,
        index_name: str,
        limit: int,
        search_type: str,
        correlation_id: str,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search (compatible with 0.1.0)."""
        try:
            # Generate query embedding based on search type
            if search_type == "semantic":
                query_vector = np.array(
                    self.embedding_service.encode_semantic(query), dtype=np.float32
                )
            elif search_type == "emotional":
                query_vector = np.array(
                    self.embedding_service.encode_emotional(query), dtype=np.float32
                )
            else:
                logger.error(
                    f"Unsupported search type: {search_type}",
                    correlation_id=correlation_id,
                )
                return []

            # Convert to binary format for Redis
            vector_blob = query_vector.tobytes()

            redis_client = await self._get_redis_client()

            # Build FT.SEARCH command with KNN (compatible with 0.1.0)
            search_cmd = [
                "FT.SEARCH",
                index_name,
                f"*=>[KNN {limit} @{vector_field} $vector AS distance]",
                "PARAMS",
                "2",
                "vector",
                vector_blob,
                "RETURN",
                "6",
                "story_id",
                "title",
                "participants",
                "tags",
                "outcome",
                "distance",
                "SORTBY",
                "distance",
                "ASC",
                "DIALECT",
                "2",
            ]

            # Execute search
            logger.debug(
                "Executing vector search",
                search_type=search_type,
                correlation_id=correlation_id,
            )
            result = await redis_client.execute_command(*search_cmd)

            # Parse results - Redis returns: [total_count, doc1_id, doc1_fields, doc2_id, doc2_fields, ...]
            if not result or len(result) < 2:
                return []

            total_count = result[0]
            if total_count == 0:
                return []

            results = []
            # Parse document results (skip total_count at index 0)
            for i in range(1, len(result), 2):
                if i + 1 >= len(result):
                    break

                doc_id = result[i]
                doc_fields = result[i + 1]

                # Convert field list to dict
                doc_data = {}
                for j in range(0, len(doc_fields), 2):
                    if j + 1 < len(doc_fields):
                        field_name = doc_fields[j]
                        field_value = doc_fields[j + 1]

                        # Handle bytes conversion
                        if isinstance(field_name, bytes):
                            field_name = field_name.decode("utf-8")
                        if isinstance(field_value, bytes):
                            # Skip vector fields entirely (they contain binary data)
                            if "vector" in field_name.lower():
                                continue
                            try:
                                field_value = field_value.decode("utf-8")
                            except UnicodeDecodeError:
                                logger.debug(
                                    f"Skipping binary field: {field_name}",
                                    correlation_id=correlation_id,
                                )
                                continue

                        doc_data[field_name] = field_value

                # Parse JSON fields
                if "participants" in doc_data:
                    try:
                        doc_data["participants"] = json.loads(doc_data["participants"])
                    except json.JSONDecodeError:
                        pass

                if "tags" in doc_data:
                    try:
                        doc_data["tags"] = json.loads(doc_data["tags"])
                    except json.JSONDecodeError:
                        pass

                # Add metadata
                doc_data["redis_key"] = doc_id
                doc_data["search_type"] = search_type
                doc_data["granularity"] = "story"

                # Convert distance to similarity score (Redis returns cosine distance)
                if "distance" in doc_data:
                    try:
                        distance = float(doc_data["distance"])
                        # For cosine distance, similarity = 1 - distance (assuming normalized vectors)
                        similarity = max(0, 1 - distance)
                        doc_data["similarity_score"] = round(similarity, 4)
                    except (ValueError, TypeError):
                        doc_data["similarity_score"] = 0.0

                # Clean up any remaining bytes objects
                clean_data = {}
                for k, v in doc_data.items():
                    if isinstance(v, bytes):
                        logger.debug(
                            f"Skipping bytes field: {k}", correlation_id=correlation_id
                        )
                        continue
                    clean_data[k] = v

                results.append(clean_data)

            logger.info(
                "Vector search completed",
                search_type=search_type,
                results_count=len(results),
                query=query[:50],
                correlation_id=correlation_id,
            )
            return results

        except Exception as e:
            logger.error(
                "Vector search failed",
                search_type=search_type,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            return []

    async def close(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None

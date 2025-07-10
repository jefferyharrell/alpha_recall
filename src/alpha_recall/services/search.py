"""Search service functions that accept pre-generated embeddings for optimal performance."""

import json
from typing import Any

import numpy as np

from ..logging import get_logger
from ..services.factory import get_narrative_service
from ..services.memgraph import get_memgraph_service
from ..services.redis import get_redis_service
from ..services.time import time_service


def search_shortterm_with_embeddings(
    semantic_embedding: np.ndarray | list[float],
    emotional_embedding: np.ndarray | list[float] | None = None,
    limit: int = 10,
    cutoff_timestamp: float = 0,
    search_type: str = "semantic",
    query: str = "",
) -> list[dict[str, Any]]:
    """Search short-term memories using pre-generated embeddings.

    Args:
        semantic_embedding: Pre-generated semantic embedding vector
        emotional_embedding: Pre-generated emotional embedding vector (optional)
        limit: Maximum number of memories to return
        cutoff_timestamp: Filter memories newer than this timestamp
        search_type: "semantic" or "emotional"
        query: Original query string (for emotional search fallback)

    Returns:
        List of memory dictionaries with scores and metadata
    """
    logger = get_logger("services.search.shortterm")

    # Get Redis service
    redis_service = get_redis_service()
    client = redis_service.client

    # Check if we have any memories to search
    memory_count = client.zcard("memory_index")
    if memory_count == 0:
        logger.info("No memories found in index")
        return []

    if search_type == "semantic":
        return _search_stm_semantic_with_embedding(
            redis_service, semantic_embedding, limit, cutoff_timestamp, logger
        )
    else:
        # For emotional search, we'd use emotional_embedding here
        # For now, fall back to text matching
        return _search_stm_emotional_fallback(
            client, query, limit, cutoff_timestamp, logger
        )


def search_longterm_with_embeddings(
    semantic_embedding: np.ndarray | list[float],
    emotional_embedding: np.ndarray | list[float] | None = None,
    entity: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search long-term memory observations using pre-generated embeddings.

    Args:
        semantic_embedding: Pre-generated semantic embedding vector
        emotional_embedding: Pre-generated emotional embedding vector (optional)
        entity: Optional entity name to filter results
        limit: Maximum number of observations to return

    Returns:
        List of observation dictionaries with scores and metadata
    """
    logger = get_logger("services.search.longterm")

    # Get Memgraph service
    memgraph_service = get_memgraph_service()

    try:
        # Convert embeddings to lists for Memgraph query parameters
        semantic_vector = (
            semantic_embedding.tolist()
            if isinstance(semantic_embedding, np.ndarray)
            else semantic_embedding
        )
        emotional_vector = (
            emotional_embedding.tolist()
            if isinstance(emotional_embedding, np.ndarray)
            else emotional_embedding
        )

        all_results = []

        # Semantic search
        semantic_query = """
        CALL vector_search.search("semantic_vector_index", $limit, $query_vector)
        YIELD node, similarity
        MATCH (e:Entity)-[:HAS_OBSERVATION]->(node)
        """

        semantic_params = {
            "limit": limit * 2,  # Get more results since we'll merge and filter
            "query_vector": semantic_vector,
        }

        if entity:
            semantic_query += " WHERE e.name = $entity_filter"
            semantic_params["entity_filter"] = entity

        semantic_query += """
        RETURN e.name as entity_name,
               node.content as observation,
               node.created_at as created_at,
               similarity,
               "semantic" as search_type
        ORDER BY similarity DESC
        """

        semantic_results = list(
            memgraph_service.db.execute_and_fetch(semantic_query, semantic_params)
        )

        # Process semantic results
        for row in semantic_results:
            all_results.append(
                {
                    "entity_name": row["entity_name"],
                    "observation": row["observation"],
                    "created_at": row["created_at"],
                    "similarity_score": float(row["similarity"]),
                    "search_type": row["search_type"],
                }
            )

        # Emotional search (if emotional embedding is provided)
        if emotional_vector:
            emotional_query = """
            CALL vector_search.search("emotional_vector_index", $limit, $query_vector)
            YIELD node, similarity
            MATCH (e:Entity)-[:HAS_OBSERVATION]->(node)
            """

            emotional_params = {
                "limit": limit * 2,  # Get more results since we'll merge and filter
                "query_vector": emotional_vector,
            }

            if entity:
                emotional_query += " WHERE e.name = $entity_filter"
                emotional_params["entity_filter"] = entity

            emotional_query += """
            RETURN e.name as entity_name,
                   node.content as observation,
                   node.created_at as created_at,
                   similarity,
                   "emotional" as search_type
            ORDER BY similarity DESC
            """

            emotional_results = list(
                memgraph_service.db.execute_and_fetch(emotional_query, emotional_params)
            )

            # Process emotional results
            for row in emotional_results:
                all_results.append(
                    {
                        "entity_name": row["entity_name"],
                        "observation": row["observation"],
                        "created_at": row["created_at"],
                        "similarity_score": float(row["similarity"]),
                        "search_type": row["search_type"],
                    }
                )

        # Filter by similarity threshold (0.3 like STM)
        similarity_threshold = 0.3
        filtered_results = [
            r for r in all_results if r["similarity_score"] >= similarity_threshold
        ]

        # Merge and deduplicate results
        seen = set()
        unique_results = []

        for result in filtered_results:
            # Create a unique key based on entity and observation content
            key = (result["entity_name"], result["observation"])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        # Sort by similarity score (highest first) and limit
        unique_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        observations = unique_results[:limit]

        logger.info(
            "LTM search completed with pre-generated embeddings",
            entity_filter=entity,
            final_results=len(observations),
        )

        return observations

    except Exception as e:
        logger.error(f"LTM search with pre-generated embeddings failed: {e}")
        return []


async def search_narratives_with_embeddings(
    semantic_embedding: np.ndarray | list[float],
    emotional_embedding: np.ndarray | list[float] | None = None,
    search_type: str = "semantic",
    granularity: str = "story",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search narrative memories using pre-generated embeddings.

    Args:
        semantic_embedding: Pre-generated semantic embedding vector
        emotional_embedding: Pre-generated emotional embedding vector (optional)
        search_type: "semantic", "emotional", or "both"
        granularity: "story", "paragraph", or "both"
        limit: Maximum number of results to return

    Returns:
        List of narrative result dictionaries with scores and metadata
    """
    logger = get_logger("services.search.narratives")

    # Get narrative service
    narrative_service = get_narrative_service()

    # Import redis async client

    try:
        results = []

        # Get Redis client
        redis_client = await narrative_service._get_redis_client()

        if search_type in ["semantic", "both"]:
            if granularity in ["story", "both"]:
                # Story-level semantic search with pre-generated embedding
                story_results = await _narrative_vector_search_with_embedding(
                    redis_client=redis_client,
                    query_embedding=semantic_embedding,
                    vector_field="full_semantic_vector",
                    index_name="idx:narrative_stories",
                    limit=limit,
                    search_type="semantic",
                    logger=logger,
                )
                results.extend(story_results)

        if search_type in ["emotional", "both"] and emotional_embedding is not None:
            if granularity in ["story", "both"]:
                # Story-level emotional search with pre-generated embedding
                story_results = await _narrative_vector_search_with_embedding(
                    redis_client=redis_client,
                    query_embedding=emotional_embedding,
                    vector_field="full_emotional_vector",
                    index_name="idx:narrative_stories",
                    limit=limit,
                    search_type="emotional",
                    logger=logger,
                )
                results.extend(story_results)

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
            "Narrative search completed with pre-generated embeddings",
            results_count=len(unique_results),
            search_type=search_type,
            granularity=granularity,
        )

        return unique_results

    except Exception as e:
        logger.error(f"Narrative search with pre-generated embeddings failed: {e}")
        return []


async def _narrative_vector_search_with_embedding(
    redis_client,
    query_embedding: np.ndarray | list[float],
    vector_field: str,
    index_name: str,
    limit: int,
    search_type: str,
    logger,
) -> list[dict[str, Any]]:
    """Perform vector similarity search with pre-generated embedding."""
    try:
        # Convert embedding to binary format for Redis
        if isinstance(query_embedding, np.ndarray):
            query_vector = query_embedding.astype(np.float32)
        else:
            query_vector = np.array(query_embedding, dtype=np.float32)

        vector_blob = query_vector.tobytes()

        # Build FT.SEARCH command with KNN
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
                        # Skip vector fields (they contain binary data)
                        if "vector" in field_name.lower():
                            continue
                        try:
                            field_value = field_value.decode("utf-8")
                        except UnicodeDecodeError:
                            continue

                    doc_data[field_name] = field_value

            # Convert distance to similarity score (1 - distance)
            distance = float(doc_data.get("distance", 1.0))
            similarity_score = 1.0 - distance

            # Parse participants and tags from JSON strings
            participants = []
            tags = []

            try:
                participants_str = doc_data.get("participants", "[]")
                if participants_str:
                    participants = json.loads(participants_str)
            except (json.JSONDecodeError, TypeError):
                participants = []

            try:
                tags_str = doc_data.get("tags", "[]")
                if tags_str:
                    tags = json.loads(tags_str)
            except (json.JSONDecodeError, TypeError):
                tags = []

            results.append(
                {
                    "story_id": doc_data.get("story_id"),
                    "title": doc_data.get("title"),
                    "participants": participants,
                    "tags": tags,
                    "outcome": doc_data.get("outcome"),
                    "similarity_score": similarity_score,
                    "search_type": search_type,
                    "granularity": "story",
                    "content": doc_data.get(
                        "title"
                    ),  # Use title as content for compatibility
                }
            )

        return results

    except Exception as e:
        logger.error(
            f"Narrative vector search with pre-generated embedding failed: {e}"
        )
        return []


def _search_stm_semantic_with_embedding(
    redis_service,
    query_embedding: np.ndarray | list[float],
    limit: int,
    cutoff_timestamp: float,
    logger,
) -> list[dict[str, Any]]:
    """Perform semantic vector search using pre-generated embedding."""

    # Ensure vector index exists
    if not redis_service.ensure_vector_index_exists():
        logger.error("Vector index unavailable and could not be created")
        return []

    # Convert embedding to binary format for Redis vector search
    if isinstance(query_embedding, np.ndarray):
        vector_bytes = query_embedding.astype(np.float32).tobytes()
    else:
        vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

    # Build the vector search query
    search_query = (
        f"*=>[KNN {limit} @semantic_vector $query_vector AS similarity_score]"
    )

    try:
        # Execute vector search
        search_result = redis_service.client.execute_command(
            "FT.SEARCH",
            "memory_semantic_index",
            search_query,
            "PARAMS",
            "2",
            "query_vector",
            vector_bytes,
            "SORTBY",
            "similarity_score",
            "ASC",  # Lower scores = more similar in cosine distance
            "RETURN",
            "4",
            "content",
            "created_at",
            "id",
            "similarity_score",
            "DIALECT",
            "2",
        )

        # Parse search results
        total_results = search_result[0]
        logger.info(f"Vector search returned {total_results} results")

        memories = []
        now = time_service.utc_now()

        # Results come in pairs: [doc_key, [field1, value1, field2, value2, ...]]
        for i in range(1, len(search_result), 2):
            doc_key = search_result[i].decode("utf-8")
            doc_fields = search_result[i + 1]

            # Extract memory ID from document key (format: "memory:id")
            memory_id = doc_key.replace("memory:", "")

            # Parse field values (they come as [field, value, field, value, ...])
            fields = {}
            for j in range(0, len(doc_fields), 2):
                field_name = doc_fields[j].decode("utf-8")
                field_value = doc_fields[j + 1].decode("utf-8")
                fields[field_name] = field_value

            # Check time filter if specified
            if cutoff_timestamp > 0:
                try:
                    created_at = fields.get("created_at", "")
                    memory_timestamp = time_service.parse(created_at).timestamp()
                    if memory_timestamp < cutoff_timestamp:
                        continue
                except Exception:
                    # Skip if we can't parse the timestamp
                    continue

            # Convert similarity score from distance to similarity (1 - distance)
            distance_score = float(fields.get("similarity_score", 1.0))
            similarity_score = 1.0 - distance_score

            # Calculate human-readable age
            try:
                created_time = time_service.parse(fields.get("created_at", ""))
                age = created_time.diff_for_humans(now)
            except Exception:
                age = "unknown"

            memories.append(
                {
                    "id": fields.get("id", memory_id),
                    "content": fields.get("content", ""),
                    "created_at": fields.get("created_at", ""),
                    "age": age,
                    "similarity_score": similarity_score,
                    "search_type": "semantic",
                }
            )

        return memories[:limit]  # Ensure we don't exceed the requested limit

    except Exception as search_error:
        logger.error(f"Semantic vector search failed: {search_error}")
        return []


def _search_stm_emotional_fallback(
    client, query: str, limit: int, cutoff_timestamp: float, logger
) -> list[dict[str, Any]]:
    """Fallback emotional search using text matching."""
    logger.info("Using text-based fallback for emotional search")

    # Get all memory IDs within time range
    if cutoff_timestamp > 0:
        memory_ids_with_scores = client.zrangebyscore(
            "memory_index", cutoff_timestamp, "+inf", withscores=True
        )
    else:
        memory_ids_with_scores = client.zrevrange(
            "memory_index", 0, -1, withscores=True
        )

    memories = []
    now = time_service.utc_now()
    query_lower = query.lower()

    # Search through memories for text matches
    for memory_id_bytes, _timestamp in memory_ids_with_scores:
        memory_id = memory_id_bytes.decode("utf-8")
        memory_key = f"memory:{memory_id}"

        # Get memory data from hash
        memory_data = client.hmget(memory_key, ["content", "created_at", "id"])

        if memory_data[0] is not None:  # Content exists
            content = memory_data[0].decode("utf-8")
            created_at = memory_data[1].decode("utf-8") if memory_data[1] else ""
            stored_id = memory_data[2].decode("utf-8") if memory_data[2] else memory_id

            # Simple text matching (case-insensitive)
            if query_lower in content.lower():
                # Calculate human-readable age
                try:
                    created_time = time_service.parse(created_at)
                    age = created_time.diff_for_humans(now)
                except Exception:
                    age = "unknown"

                # Simple relevance score based on query frequency in content
                content_lower = content.lower()
                relevance_score = content_lower.count(query_lower) / len(
                    content_lower.split()
                )

                memories.append(
                    {
                        "id": stored_id,
                        "content": content,
                        "created_at": created_at,
                        "age": age,
                        "relevance_score": relevance_score,
                        "search_type": "emotional_fallback",
                    }
                )

        # Stop if we have enough results
        if len(memories) >= limit:
            break

    # Sort by relevance score (descending)
    memories.sort(key=lambda m: m.get("relevance_score", 0), reverse=True)
    return memories[:limit]

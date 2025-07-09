"""Search narratives tool for vector similarity search across stories."""

import asyncio
import json
import time

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.factory import get_narrative_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["search_narratives", "register_search_narratives_tools"]


def search_narratives(
    query: str,
    search_type: str = "semantic",
    granularity: str = "story",
    limit: int = 10,
) -> str:
    """Search narrative memories using vector similarity.

    Args:
        query: The search query text
        search_type: "semantic", "emotional", or "both" (default: "semantic")
        granularity: "story", "paragraph", or "both" (default: "story")
        limit: Maximum number of results to return (default: 10)

    Returns:
        JSON string with search results
    """
    correlation_id = generate_correlation_id("search_narratives")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.search_narratives")
    logger.info(
        "Search narratives request received",
        query=query,
        search_type=search_type,
        granularity=granularity,
        limit=limit,
        correlation_id=correlation_id,
    )

    try:
        # Validate parameters
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if search_type not in ["semantic", "emotional", "both"]:
            raise ValueError("search_type must be 'semantic', 'emotional', or 'both'")

        if granularity not in ["story", "paragraph", "both"]:
            raise ValueError("granularity must be 'story', 'paragraph', or 'both'")

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        # Perform the actual search using NarrativeService
        start_time = time.time()
        narrative_service = get_narrative_service()

        # Handle async call properly (avoid nested event loops)
        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    narrative_service.search_stories(
                        query=query,
                        search_type=search_type,
                        granularity=granularity,
                        limit=limit,
                    ),
                )
                results = future.result()
        except RuntimeError:
            # No existing event loop, safe to use asyncio.run
            results = asyncio.run(
                narrative_service.search_stories(
                    query=query,
                    search_type=search_type,
                    granularity=granularity,
                    limit=limit,
                )
            )
        search_time_ms = int((time.time() - start_time) * 1000)

        # Return response structure
        response = {
            "success": True,
            "search": {
                "query": query,
                "search_type": search_type,
                "granularity": granularity,
                "limit": limit,
            },
            "results": results,
            "metadata": {
                "results_count": len(results),
                "search_time_ms": search_time_ms,
                "search_method": "vector_similarity",
                "embedding_model": "dual_semantic_emotional",
            },
            "correlation_id": correlation_id,
        }

        logger.info(
            "Narrative search completed",
            query=query,
            results_count=0,
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(
            "Failed to search narratives",
            query=query,
            error=str(e),
            error_type=type(e).__name__,
            correlation_id=correlation_id,
        )

        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "correlation_id": correlation_id,
        }

        return json.dumps(error_response, indent=2)


def register_search_narratives_tools(mcp: FastMCP) -> None:
    """Register search_narratives tools with the MCP server."""
    logger = get_logger("tools.search_narratives")

    # Register the search_narratives tool
    mcp.tool(search_narratives)

    logger.debug("Search narratives tools registered")

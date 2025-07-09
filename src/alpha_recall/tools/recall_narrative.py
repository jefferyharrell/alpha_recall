"""Recall narrative tool for retrieving complete stories by ID."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["recall_narrative", "register_recall_narrative_tools"]


def recall_narrative(story_id: str) -> str:
    """Retrieve a complete narrative story by its story_id.

    Args:
        story_id: The unique identifier for the story to retrieve

    Returns:
        JSON string with complete story data
    """
    correlation_id = generate_correlation_id("recall_narrative")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.recall_narrative")
    logger.info(
        "Recall narrative request received",
        story_id=story_id,
        correlation_id=correlation_id,
    )

    try:
        # Validate parameters
        if not story_id.strip():
            raise ValueError("story_id cannot be empty")

        # Validate story_id format
        if not story_id.startswith("story_"):
            raise ValueError("story_id must start with 'story_'")

        # TODO: Implement actual retrieval
        # This is a placeholder until we implement the NarrativeService
        logger.info(
            "Narrative recall placeholder",
            story_id=story_id,
            correlation_id=correlation_id,
        )

        # Return response structure
        response = {
            "success": True,
            "story": {
                "story_id": story_id,
                "title": "Placeholder Story",
                "created_at": "2025-07-09T14:30:00.000000Z",
                "participants": ["Alpha", "Jeffery"],
                "tags": ["placeholder", "implementation"],
                "outcome": "ongoing",
                "references": [],
                "paragraphs": [
                    {
                        "order": 0,
                        "content": "This is a placeholder story that would be retrieved from storage.",
                    }
                ],
                "metadata": {
                    "paragraph_count": 1,
                    "word_count": 12,
                    "storage_location": "hybrid_redis_memgraph",
                },
            },
            "correlation_id": correlation_id,
        }

        logger.info(
            "Narrative recall completed",
            story_id=story_id,
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(
            "Failed to recall narrative",
            story_id=story_id,
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


def register_recall_narrative_tools(mcp: FastMCP) -> None:
    """Register recall_narrative tools with the MCP server."""
    logger = get_logger("tools.recall_narrative")

    # Register the recall_narrative tool
    mcp.tool(recall_narrative)

    logger.debug("Recall narrative tools registered")

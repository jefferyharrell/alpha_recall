"""Recall narrative tool for retrieving complete stories by ID."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.factory import get_narrative_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["recall_narrative", "register_recall_narrative_tools"]


async def recall_narrative(story_id: str) -> str:
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

        # Retrieve the story using NarrativeService
        narrative_service = get_narrative_service()

        # Call async service method
        story_data = await narrative_service.get_story(story_id)

        if not story_data:
            # Story not found
            error_response = {
                "success": False,
                "error": f"Story with ID '{story_id}' not found",
                "error_type": "StoryNotFound",
                "correlation_id": correlation_id,
            }
            return json.dumps(error_response, indent=2)

        # Return response structure
        response = {
            "success": True,
            "story": {
                "story_id": story_data["story_id"],
                "title": story_data["title"],
                "created_at": story_data["created_at"],
                "participants": story_data["participants"],
                "tags": story_data["tags"],
                "outcome": story_data["outcome"],
                "references": story_data.get("references", []),
                "paragraphs": story_data["paragraphs"],
                "metadata": {
                    "paragraph_count": len(story_data["paragraphs"]),
                    "word_count": sum(
                        len(p["text"].split()) for p in story_data["paragraphs"]
                    ),
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

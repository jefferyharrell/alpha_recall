"""Remember narrative tool for storing experiential stories."""

import json
import time

from fastmcp import FastMCP

from ..logging import get_logger
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["remember_narrative", "register_remember_narrative_tools"]


def remember_narrative(
    title: str,
    paragraphs: list[str],
    participants: list[str],
    outcome: str = "ongoing",
    tags: list[str] | None = None,
    references: list[str] | None = None,
) -> str:
    """Store a narrative memory with hybrid storage.

    Args:
        title: The title of the narrative story
        paragraphs: List of paragraphs that make up the story
        participants: List of participants involved in the story
        outcome: Story outcome status (ongoing, breakthrough, resolution, etc.)
        tags: Optional list of tags/topics for the story
        references: Optional list of references to other stories

    Returns:
        JSON string with storage confirmation and story details
    """
    correlation_id = generate_correlation_id("narrative")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.remember_narrative")
    logger.info(
        "Remember narrative request received",
        title=title,
        paragraph_count=len(paragraphs),
        participants=participants,
        outcome=outcome,
        tags=tags or [],
        references=references or [],
        correlation_id=correlation_id,
    )

    try:
        # Generate story ID with timestamp
        story_id = f"story_{int(time.time())}_{correlation_id.split('_')[1]}"

        # Clean up data first
        clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]
        clean_participants = [p.strip() for p in participants if p.strip()]
        clean_tags = [t.strip() for t in (tags or []) if t.strip()]
        clean_references = [r.strip() for r in (references or []) if r.strip()]

        # Validate required parameters (after cleaning)
        if not title.strip():
            raise ValueError("Title cannot be empty")
        if not clean_paragraphs:
            raise ValueError("At least one non-empty paragraph required")
        if not clean_participants:
            raise ValueError("At least one participant required")

        current_time = time.time()
        iso_timestamp = time.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time)
        )

        # TODO: Implement actual storage
        # This is a placeholder until we implement the NarrativeService
        logger.info(
            "Narrative storage placeholder",
            story_id=story_id,
            title=title,
            paragraph_count=len(clean_paragraphs),
            correlation_id=correlation_id,
        )

        # Return response structure
        response = {
            "success": True,
            "story": {
                "story_id": story_id,
                "title": title,
                "created_at": iso_timestamp,
                "participants": clean_participants,
                "tags": clean_tags,
                "outcome": outcome,
                "paragraph_count": len(clean_paragraphs),
                "references": clean_references,
            },
            "processing": {
                "paragraphs_processed": len(clean_paragraphs),
                "embeddings_generated": "pending",  # Will be actual count when implemented
                "storage_location": "hybrid_redis_memgraph",
            },
            "correlation_id": correlation_id,
        }

        logger.info(
            "Narrative storage completed",
            story_id=story_id,
            success=True,
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(
            "Failed to store narrative",
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


def register_remember_narrative_tools(mcp: FastMCP) -> None:
    """Register remember_narrative tools with the MCP server."""
    logger = get_logger("tools.remember_narrative")

    # Register the remember_narrative tool
    mcp.tool(remember_narrative)

    logger.debug("Remember narrative tools registered")

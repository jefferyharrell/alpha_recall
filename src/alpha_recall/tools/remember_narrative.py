"""Remember narrative tool for storing experiential stories."""

import asyncio
import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.factory import get_narrative_service
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

        # Get narrative service and store the story
        narrative_service = get_narrative_service()

        # Handle async call properly (avoid nested event loops)
        try:
            asyncio.get_running_loop()
            # We're in an existing event loop, create a new task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    narrative_service.store_story(
                        title=title.strip(),
                        paragraphs=clean_paragraphs,
                        participants=clean_participants,
                        tags=clean_tags,
                        outcome=outcome,
                        references=clean_references,
                    ),
                )
                result = future.result()
        except RuntimeError:
            # No existing event loop, safe to use asyncio.run
            result = asyncio.run(
                narrative_service.store_story(
                    title=title.strip(),
                    paragraphs=clean_paragraphs,
                    participants=clean_participants,
                    tags=clean_tags,
                    outcome=outcome,
                    references=clean_references,
                )
            )

        # If storage failed, return error response
        if not result.get("success", False):
            error_response = {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "error_type": result.get("error_type", "UnknownError"),
                "correlation_id": correlation_id,
            }
            return json.dumps(error_response, indent=2)

        # Return successful response
        response = {
            "success": True,
            "story": {
                "story_id": result["story_id"],
                "title": result["title"],
                "created_at": result["created_at"],
                "participants": clean_participants,
                "tags": clean_tags,
                "outcome": outcome,
                "paragraph_count": result["paragraph_count"],
                "references": clean_references,
            },
            "processing": {
                "paragraphs_processed": result["paragraph_count"],
                "embeddings_generated": result["embeddings_generated"],
                "storage_location": result["storage_location"],
            },
            "correlation_id": correlation_id,
        }

        logger.info(
            "Narrative storage completed",
            story_id=result["story_id"],
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

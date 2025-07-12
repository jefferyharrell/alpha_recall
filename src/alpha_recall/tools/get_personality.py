"""Get complete personality structure for introspection."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..services.time import time_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["get_personality", "register_get_personality_tools"]

logger = get_logger("tools.get_personality")


def get_personality() -> str:
    """Get complete personality structure for introspection.

    Returns the full personality graph with all traits and their directives,
    providing a comprehensive view for personality discovery and analysis.

    Returns:
        JSON string containing the complete personality structure
    """
    correlation_id = generate_correlation_id("get_personality")
    set_correlation_id(correlation_id)

    logger.info(
        "Starting complete personality retrieval", correlation_id=correlation_id
    )

    try:
        memgraph_service = get_memgraph_service()

        # Query to get all personality traits and their directives
        query = """
        MATCH (agent:Agent_Personality)-[:HAS_TRAIT]->(trait:Personality_Trait)
        OPTIONAL MATCH (trait)-[:HAS_DIRECTIVE]->(directive:Personality_Directive)
        RETURN trait.name as trait_name,
               trait.description as trait_description,
               trait.weight as trait_weight,
               trait.created_at as trait_created_at,
               trait.last_updated as trait_last_updated,
               directive.instruction as directive_instruction,
               directive.weight as directive_weight,
               directive.created_at as directive_created_at
        ORDER BY trait.name, directive.weight DESC
        """

        logger.debug(
            "Executing personality retrieval query", correlation_id=correlation_id
        )
        results = list(memgraph_service.db.execute_and_fetch(query))

        if not results:
            logger.info("No personality traits found", correlation_id=correlation_id)
            return json.dumps(
                {
                    "success": True,
                    "personality": {},
                    "trait_count": 0,
                    "directive_count": 0,
                    "retrieved_at": time_service.to_utc_isoformat(),
                    "correlation_id": correlation_id,
                }
            )

        # Build personality structure from explicit field results
        personality = {}
        total_directives = 0

        # Group results by trait_name since we now get one row per directive
        traits_data = {}
        for row in results:
            trait_name = row["trait_name"]

            if trait_name not in traits_data:
                # Handle datetime serialization with pendulum compatibility
                trait_created_at = row["trait_created_at"]
                trait_last_updated = row["trait_last_updated"]

                # Convert standard datetime objects to pendulum if needed
                if (
                    trait_created_at
                    and hasattr(trait_created_at, "year")
                    and not hasattr(trait_created_at, "in_timezone")
                ):
                    import pendulum

                    trait_created_at = pendulum.instance(trait_created_at)
                if (
                    trait_last_updated
                    and hasattr(trait_last_updated, "year")
                    and not hasattr(trait_last_updated, "in_timezone")
                ):
                    import pendulum

                    trait_last_updated = pendulum.instance(trait_last_updated)

                trait_created_at_str = time_service.to_utc_isoformat(trait_created_at)
                trait_last_updated_str = time_service.to_utc_isoformat(
                    trait_last_updated
                )

                traits_data[trait_name] = {
                    "description": row["trait_description"],
                    "weight": row["trait_weight"],
                    "created_at": trait_created_at_str,
                    "last_updated": trait_last_updated_str,
                    "directives": [],
                }

            # Add directive if it exists (directive fields will be null for traits without directives)
            if row["directive_instruction"] is not None:
                directive_created_at = row["directive_created_at"]

                # Convert standard datetime objects to pendulum if needed
                if (
                    directive_created_at
                    and hasattr(directive_created_at, "year")
                    and not hasattr(directive_created_at, "in_timezone")
                ):
                    import pendulum

                    directive_created_at = pendulum.instance(directive_created_at)

                directive_created_at_str = time_service.to_utc_isoformat(
                    directive_created_at
                )

                traits_data[trait_name]["directives"].append(
                    {
                        "instruction": row["directive_instruction"],
                        "weight": row["directive_weight"],
                        "created_at": directive_created_at_str,
                    }
                )
                total_directives += 1

        personality = traits_data

        logger.info(
            "Successfully retrieved complete personality",
            correlation_id=correlation_id,
            trait_count=len(personality),
            directive_count=total_directives,
        )

        return json.dumps(
            {
                "success": True,
                "personality": personality,
                "trait_count": len(personality),
                "directive_count": total_directives,
                "retrieved_at": time_service.to_utc_isoformat(),
                "correlation_id": correlation_id,
            }
        )

    except Exception as e:
        logger.error(
            "Failed to retrieve personality",
            correlation_id=correlation_id,
            error=str(e),
            error_type=type(e).__name__,
        )

        return json.dumps(
            {
                "success": False,
                "error": f"Failed to retrieve personality: {str(e)}",
                "correlation_id": correlation_id,
            }
        )


def register_get_personality_tools(mcp: FastMCP) -> None:
    """Register this module's tools with the MCP server."""
    logger = get_logger("tools.get_personality")

    # Register tools defined at module level
    mcp.tool(get_personality)

    logger.debug("Tools registered")

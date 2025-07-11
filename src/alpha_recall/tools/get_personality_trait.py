"""Get personality trait tool for Alpha-Recall."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["get_personality_trait", "register_get_personality_trait_tools"]


def get_personality_trait(trait_name: str) -> str:
    """
    Get detailed information about a specific personality trait and its directives.

    Args:
        trait_name: The name of the personality trait to retrieve (e.g., "warmth", "intellectual_engagement")

    Returns:
        JSON string containing trait details and all associated directives
    """
    correlation_id = generate_correlation_id("get_trait")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.get_personality_trait")
    logger.info(
        f"Getting personality trait: {trait_name}", correlation_id=correlation_id
    )

    try:
        memgraph_service = get_memgraph_service()

        # Cypher query to find specific trait and its directives
        # This demonstrates pattern matching: we're looking for the path from
        # any Agent_Personality through HAS_TRAIT relationship to our specific trait,
        # then through HAS_DIRECTIVE relationships to get all associated directives
        query = """
        MATCH (root:Agent_Personality)-[:HAS_TRAIT]->(trait:Personality_Trait {name: $trait_name})-[:HAS_DIRECTIVE]->(directive:Personality_Directive)
        RETURN trait.name as trait_name,
               trait.description as trait_description,
               trait.weight as trait_weight,
               trait.created_at as trait_created_at,
               trait.last_updated as trait_last_updated,
               directive.instruction as directive_instruction,
               directive.weight as directive_weight,
               directive.created_at as directive_created_at
        ORDER BY directive.weight DESC
        """

        # Execute with parameter binding (prevents injection, good practice)
        result = list(
            memgraph_service.db.execute_and_fetch(query, {"trait_name": trait_name})
        )

        if not result:
            # Trait doesn't exist
            response = {
                "success": False,
                "error": f"Personality trait '{trait_name}' not found",
                "available_traits": [],
            }

            # Get list of available traits to help user
            available_query = """
            MATCH (trait:Personality_Trait)
            RETURN trait.name as name
            ORDER BY trait.name
            """
            available_result = list(
                memgraph_service.db.execute_and_fetch(available_query)
            )
            response["available_traits"] = [row["name"] for row in available_result]

            logger.warning(
                f"Trait '{trait_name}' not found", correlation_id=correlation_id
            )
            return json.dumps(response, indent=2)

        # Build response from query results
        # Since all rows have the same trait info, we can use the first row for trait details
        first_row = result[0]
        trait_info = {
            "success": True,
            "trait": {
                "name": first_row["trait_name"],
                "description": first_row["trait_description"],
                "weight": first_row["trait_weight"],
                "created_at": first_row["trait_created_at"],
                "last_updated": first_row["trait_last_updated"],
                "directives": [],
            },
        }

        # Add all directives (already ordered by weight DESC)
        for row in result:
            trait_info["trait"]["directives"].append(
                {
                    "instruction": row["directive_instruction"],
                    "weight": row["directive_weight"],
                    "created_at": row["directive_created_at"],
                }
            )

        directive_count = len(trait_info["trait"]["directives"])
        logger.info(
            f"Retrieved trait '{trait_name}' with {directive_count} directives",
            correlation_id=correlation_id,
        )

        return json.dumps(trait_info, indent=2)

    except Exception as e:
        logger.error(
            f"Error retrieving personality trait: {e}", correlation_id=correlation_id
        )
        error_response = {
            "success": False,
            "error": f"Error retrieving personality trait: {e}",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)


def register_get_personality_trait_tools(mcp: FastMCP) -> None:
    """Register get_personality_trait tools with the MCP server."""
    logger = get_logger("tools.get_personality_trait")

    # Register the tool
    mcp.tool(get_personality_trait)

    logger.debug("get_personality_trait tools registered")

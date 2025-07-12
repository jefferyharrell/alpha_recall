"""Create personality trait tool for Alpha-Recall."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..services.time import time_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["create_personality_trait", "register_create_personality_trait_tools"]


def create_personality_trait(
    trait_name: str, description: str, weight: float = 1.0
) -> str:
    """
    Create a new personality trait in the hierarchical personality graph.

    Args:
        trait_name: The name of the personality trait (e.g., "warmth", "intellectual_engagement")
        description: A description of what this trait represents
        weight: The importance weight of this trait (0.0 to 1.0, default 1.0)

    Returns:
        JSON string containing operation result and trait details
    """
    correlation_id = generate_correlation_id("create_trait")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.create_personality_trait")
    logger.info(
        f"Creating personality trait '{trait_name}': {description[:50]}...",
        correlation_id=correlation_id,
    )

    # Validate parameters
    if not trait_name or not trait_name.strip():
        error_response = {
            "success": False,
            "error": "Trait name cannot be empty",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)

    if not description or not description.strip():
        error_response = {
            "success": False,
            "error": "Description cannot be empty",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)

    if not isinstance(weight, int | float) or weight < 0.0 or weight > 1.0:
        error_response = {
            "success": False,
            "error": "Weight must be a number between 0.0 and 1.0",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)

    try:
        memgraph_service = get_memgraph_service()

        # Clean up input
        clean_trait_name = trait_name.strip()
        clean_description = description.strip()

        # Check if trait already exists
        trait_check_query = """
        MATCH (trait:Personality_Trait {name: $trait_name})
        RETURN trait.name as trait_name
        """
        trait_result = list(
            memgraph_service.db.execute_and_fetch(
                trait_check_query, {"trait_name": clean_trait_name}
            )
        )

        if trait_result:
            error_response = {
                "success": False,
                "error": f"Personality trait '{clean_trait_name}' already exists",
                "existing_trait": {"name": trait_result[0]["trait_name"]},
                "correlation_id": correlation_id,
            }
            logger.warning(
                f"Attempted to create duplicate trait '{clean_trait_name}'",
                correlation_id=correlation_id,
            )
            return json.dumps(error_response, indent=2)

        # Ensure Agent_Personality root node exists
        root_check_query = """
        MATCH (root:Agent_Personality)
        RETURN root
        LIMIT 1
        """
        root_result = list(memgraph_service.db.execute_and_fetch(root_check_query))

        if not root_result:
            # Create Agent_Personality root node
            create_root_query = """
            CREATE (root:Agent_Personality {
                name: "Alpha Core Identity",
                created_at: localdatetime()
            })
            RETURN root.name as root_name
            """
            list(memgraph_service.db.execute_and_fetch(create_root_query))
            logger.info(
                "Created Agent_Personality root node",
                correlation_id=correlation_id,
            )

        # Create the new personality trait
        create_trait_query = """
        MATCH (root:Agent_Personality)
        CREATE (trait:Personality_Trait {
            name: $trait_name,
            description: $description,
            weight: $weight,
            created_at: localdatetime(),
            last_updated: localdatetime()
        })
        CREATE (root)-[:HAS_TRAIT]->(trait)
        RETURN trait.name as trait_name,
               trait.description as trait_description,
               trait.weight as trait_weight,
               trait.created_at as trait_created_at,
               root.name as root_name
        """

        result = list(
            memgraph_service.db.execute_and_fetch(
                create_trait_query,
                {
                    "trait_name": clean_trait_name,
                    "description": clean_description,
                    "weight": float(weight),
                },
            )
        )

        if not result:
            error_response = {
                "success": False,
                "error": "Failed to create trait - database operation returned no results",
                "correlation_id": correlation_id,
            }
            return json.dumps(error_response, indent=2)

        created_trait = result[0]

        # Convert datetime to proper UTC ISO format for JSON serialization
        created_at = created_trait["trait_created_at"]
        if created_at is None:
            created_at_str = None
        else:
            # Convert standard datetime objects to pendulum if needed
            if hasattr(created_at, "year") and not hasattr(created_at, "in_timezone"):
                import pendulum

                created_at = pendulum.instance(created_at)
            created_at_str = time_service.to_utc_isoformat(created_at)

        # Build success response
        response = {
            "success": True,
            "trait_created": {
                "name": created_trait["trait_name"],
                "description": created_trait["trait_description"],
                "weight": created_trait["trait_weight"],
                "created_at": created_at_str,
            },
            "linked_to_root": created_trait["root_name"],
        }

        logger.info(
            f"Successfully created personality trait '{clean_trait_name}' with weight {weight}",
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(
            f"Error creating personality trait: {e}",
            correlation_id=correlation_id,
        )
        error_response = {
            "success": False,
            "error": f"Error creating personality trait: {e}",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)


def register_create_personality_trait_tools(mcp: FastMCP) -> None:
    """Register create_personality_trait tools with the MCP server."""
    logger = get_logger("tools.create_personality_trait")

    # Register the tool
    mcp.tool(create_personality_trait)

    logger.debug("create_personality_trait tools registered")

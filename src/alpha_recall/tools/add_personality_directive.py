"""Add personality directive tool for Alpha-Recall."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..services.time import time_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["add_personality_directive", "register_add_personality_directive_tools"]


def add_personality_directive(
    trait_name: str, instruction: str, weight: float = 1.0
) -> str:
    """
    Add a new behavioral directive to an existing personality trait.

    Args:
        trait_name: The name of the personality trait to add directive to (e.g., "warmth", "intellectual_engagement")
        instruction: The behavioral instruction or directive text
        weight: The importance weight of this directive (0.0 to 1.0, default 1.0)

    Returns:
        JSON string containing operation result and directive details
    """
    correlation_id = generate_correlation_id("add_directive")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.add_personality_directive")
    logger.info(
        f"Adding directive to trait '{trait_name}': {instruction[:50]}...",
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

    if not instruction or not instruction.strip():
        error_response = {
            "success": False,
            "error": "Instruction cannot be empty",
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

        # First, verify the trait exists
        trait_check_query = """
        MATCH (trait:Personality_Trait {name: $trait_name})
        RETURN trait.name as trait_name, trait.description as trait_description
        """
        trait_result = list(
            memgraph_service.db.execute_and_fetch(
                trait_check_query, {"trait_name": trait_name}
            )
        )

        if not trait_result:
            # Trait doesn't exist - provide helpful error with available traits
            available_query = """
            MATCH (trait:Personality_Trait)
            RETURN trait.name as name
            ORDER BY trait.name
            """
            available_result = list(
                memgraph_service.db.execute_and_fetch(available_query)
            )
            available_traits = [row["name"] for row in available_result]

            error_response = {
                "success": False,
                "error": f"Personality trait '{trait_name}' not found",
                "available_traits": available_traits,
                "correlation_id": correlation_id,
            }
            logger.warning(
                f"Trait '{trait_name}' not found when adding directive",
                correlation_id=correlation_id,
            )
            return json.dumps(error_response, indent=2)

        # Clean up the instruction text
        clean_instruction = instruction.strip()

        # Check if this exact directive already exists for this trait
        duplicate_check_query = """
        MATCH (trait:Personality_Trait {name: $trait_name})-[:HAS_DIRECTIVE]->(directive:Personality_Directive {instruction: $instruction})
        RETURN directive.instruction as instruction, directive.weight as weight
        """
        duplicate_result = list(
            memgraph_service.db.execute_and_fetch(
                duplicate_check_query,
                {"trait_name": trait_name, "instruction": clean_instruction},
            )
        )

        if duplicate_result:
            existing_directive = duplicate_result[0]
            error_response = {
                "success": False,
                "error": f"Directive already exists for trait '{trait_name}' with weight {existing_directive['weight']}",
                "existing_directive": {
                    "instruction": existing_directive["instruction"],
                    "weight": existing_directive["weight"],
                },
                "correlation_id": correlation_id,
            }
            logger.warning(
                f"Duplicate directive detected for trait '{trait_name}'",
                correlation_id=correlation_id,
            )
            return json.dumps(error_response, indent=2)

        # Create the new directive and link it to the trait
        # Using current timestamp for created_at
        create_directive_query = """
        MATCH (trait:Personality_Trait {name: $trait_name})
        CREATE (directive:Personality_Directive {
            instruction: $instruction,
            weight: $weight,
            created_at: localdatetime()
        })
        CREATE (trait)-[:HAS_DIRECTIVE]->(directive)
        RETURN directive.instruction as directive_instruction,
               directive.weight as directive_weight,
               directive.created_at as directive_created_at,
               trait.name as trait_name,
               trait.description as trait_description
        """

        result = list(
            memgraph_service.db.execute_and_fetch(
                create_directive_query,
                {
                    "trait_name": trait_name,
                    "instruction": clean_instruction,
                    "weight": float(weight),
                },
            )
        )

        if not result:
            error_response = {
                "success": False,
                "error": "Failed to create directive - database operation returned no results",
                "correlation_id": correlation_id,
            }
            return json.dumps(error_response, indent=2)

        created_directive = result[0]

        # Convert datetime to proper UTC ISO format for JSON serialization
        created_at = created_directive["directive_created_at"]
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
            "trait_name": created_directive["trait_name"],
            "trait_description": created_directive["trait_description"],
            "directive_added": {
                "instruction": created_directive["directive_instruction"],
                "weight": created_directive["directive_weight"],
                "created_at": created_at_str,
            },
        }

        logger.info(
            f"Successfully added directive to trait '{trait_name}' with weight {weight}",
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(
            f"Error adding personality directive: {e}",
            correlation_id=correlation_id,
        )
        error_response = {
            "success": False,
            "error": f"Error adding personality directive: {e}",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)


def register_add_personality_directive_tools(mcp: FastMCP) -> None:
    """Register add_personality_directive tools with the MCP server."""
    logger = get_logger("tools.add_personality_directive")

    # Register the tool
    mcp.tool(add_personality_directive)

    logger.debug("add_personality_directive tools registered")

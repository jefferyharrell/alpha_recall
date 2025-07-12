"""Update personality directive weight tool for Alpha-Recall."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.memgraph import get_memgraph_service
from ..services.time import time_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = [
    "update_personality_directive_weight",
    "register_update_personality_directive_weight_tools",
]


def update_personality_directive_weight(
    trait_name: str, instruction: str, new_weight: float
) -> str:
    """
    Update the weight of an existing personality directive.

    Args:
        trait_name: The name of the personality trait containing the directive
        instruction: The exact instruction text of the directive to update
        new_weight: The new weight value (0.0 to 1.0)

    Returns:
        JSON string containing operation result and updated directive details
    """
    correlation_id = generate_correlation_id("update_directive")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.update_personality_directive_weight")
    logger.info(
        f"Updating directive weight in trait '{trait_name}': {instruction[:50]}... -> {new_weight}",
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

    if not isinstance(new_weight, int | float) or new_weight < 0.0 or new_weight > 1.0:
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
        clean_instruction = instruction.strip()

        # Check if trait exists
        trait_check_query = """
        MATCH (trait:Personality_Trait {name: $trait_name})
        RETURN trait.name as trait_name,
               trait.description as trait_description
        """
        trait_result = list(
            memgraph_service.db.execute_and_fetch(
                trait_check_query, {"trait_name": clean_trait_name}
            )
        )

        if not trait_result:
            # Get list of available traits to help user
            available_query = """
            MATCH (trait:Personality_Trait)
            RETURN trait.name as name
            ORDER BY trait.name
            """
            available_result = list(
                memgraph_service.db.execute_and_fetch(available_query)
            )

            error_response = {
                "success": False,
                "error": f"Personality trait '{clean_trait_name}' not found",
                "available_traits": [row["name"] for row in available_result],
                "correlation_id": correlation_id,
            }
            logger.warning(
                f"Trait '{clean_trait_name}' not found",
                correlation_id=correlation_id,
            )
            return json.dumps(error_response, indent=2)

        # Find the specific directive to update
        directive_check_query = """
        MATCH (trait:Personality_Trait {name: $trait_name})-[:HAS_DIRECTIVE]->(directive:Personality_Directive {instruction: $instruction})
        RETURN directive.instruction as instruction,
               directive.weight as current_weight,
               directive.created_at as created_at
        """
        directive_result = list(
            memgraph_service.db.execute_and_fetch(
                directive_check_query,
                {"trait_name": clean_trait_name, "instruction": clean_instruction},
            )
        )

        if not directive_result:
            # Get available directives for this trait to help user
            available_directives_query = """
            MATCH (trait:Personality_Trait {name: $trait_name})-[:HAS_DIRECTIVE]->(directive:Personality_Directive)
            RETURN directive.instruction as instruction,
                   directive.weight as weight
            ORDER BY directive.weight DESC
            """
            available_directives = list(
                memgraph_service.db.execute_and_fetch(
                    available_directives_query, {"trait_name": clean_trait_name}
                )
            )

            error_response = {
                "success": False,
                "error": f"Directive with instruction '{clean_instruction}' not found in trait '{clean_trait_name}'",
                "trait_name": clean_trait_name,
                "available_directives": [
                    {
                        "instruction": row["instruction"],
                        "weight": row["weight"],
                    }
                    for row in available_directives
                ],
                "correlation_id": correlation_id,
            }
            logger.warning(
                f"Directive '{clean_instruction}' not found in trait '{clean_trait_name}'",
                correlation_id=correlation_id,
            )
            return json.dumps(error_response, indent=2)

        current_directive = directive_result[0]
        current_weight = current_directive["current_weight"]

        # Check if the weight is actually changing
        if abs(float(current_weight) - float(new_weight)) < 0.0001:
            response = {
                "success": True,
                "message": "Weight unchanged - directive already has the specified weight",
                "trait_name": clean_trait_name,
                "directive": {
                    "instruction": clean_instruction,
                    "weight": float(current_weight),
                    "created_at": str(current_directive["created_at"]),
                },
                "weight_change": 0.0,
            }
            logger.info(
                f"Weight unchanged for directive in trait '{clean_trait_name}'",
                correlation_id=correlation_id,
            )
            return json.dumps(response, indent=2)

        # Update the directive weight
        update_query = """
        MATCH (trait:Personality_Trait {name: $trait_name})-[:HAS_DIRECTIVE]->(directive:Personality_Directive {instruction: $instruction})
        SET directive.weight = $new_weight,
            directive.last_updated = localdatetime()
        RETURN directive.instruction as directive_instruction,
               directive.weight as directive_weight,
               directive.created_at as directive_created_at,
               directive.last_updated as directive_last_updated,
               trait.name as trait_name,
               trait.description as trait_description
        """

        update_result = list(
            memgraph_service.db.execute_and_fetch(
                update_query,
                {
                    "trait_name": clean_trait_name,
                    "instruction": clean_instruction,
                    "new_weight": float(new_weight),
                },
            )
        )

        if not update_result:
            error_response = {
                "success": False,
                "error": "Failed to update directive weight - database operation returned no results",
                "correlation_id": correlation_id,
            }
            return json.dumps(error_response, indent=2)

        updated_directive = update_result[0]

        # Convert datetime objects to proper UTC ISO format for JSON serialization
        created_at = updated_directive["directive_created_at"]
        if created_at is None:
            created_at_str = None
        else:
            # Convert standard datetime objects to pendulum if needed
            if hasattr(created_at, "year") and not hasattr(created_at, "in_timezone"):
                import pendulum

                created_at = pendulum.instance(created_at)
            created_at_str = time_service.to_utc_isoformat(created_at)

        last_updated = updated_directive["directive_last_updated"]
        if last_updated is None:
            last_updated_str = None
        else:
            # Convert standard datetime objects to pendulum if needed
            if hasattr(last_updated, "year") and not hasattr(
                last_updated, "in_timezone"
            ):
                import pendulum

                last_updated = pendulum.instance(last_updated)
            last_updated_str = time_service.to_utc_isoformat(last_updated)

        # Calculate weight change
        weight_change = float(new_weight) - float(current_weight)

        # Build success response
        response = {
            "success": True,
            "trait_name": updated_directive["trait_name"],
            "trait_description": updated_directive["trait_description"],
            "directive_updated": {
                "instruction": updated_directive["directive_instruction"],
                "previous_weight": float(current_weight),
                "new_weight": updated_directive["directive_weight"],
                "weight_change": weight_change,
                "created_at": created_at_str,
                "last_updated": last_updated_str,
            },
        }

        logger.info(
            f"Successfully updated directive weight in trait '{clean_trait_name}': {current_weight} -> {new_weight} (change: {weight_change:+.3f})",
            correlation_id=correlation_id,
        )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(
            f"Error updating personality directive weight: {e}",
            correlation_id=correlation_id,
        )
        error_response = {
            "success": False,
            "error": f"Error updating personality directive weight: {e}",
            "correlation_id": correlation_id,
        }
        return json.dumps(error_response, indent=2)


def register_update_personality_directive_weight_tools(mcp: FastMCP) -> None:
    """Register update_personality_directive_weight tools with the MCP server."""
    logger = get_logger("tools.update_personality_directive_weight")

    # Register the tool
    mcp.tool(update_personality_directive_weight)

    logger.debug("update_personality_directive_weight tools registered")

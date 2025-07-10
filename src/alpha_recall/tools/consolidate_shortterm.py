"""Consolidate short-term memories tool for systematic model evaluation."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..services.consolidation import consolidation_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["consolidate_shortterm", "register_consolidate_shortterm_tools"]


async def consolidate_shortterm(
    time_window: str = "24h",
    model_name: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Consolidate short-term memories with clean schema validation and model evaluation.

    This tool provides systematic evaluation of helper models for memory consolidation.
    It uses deterministic temperature (default 0.0) for consistent testing and returns
    detailed validation results for model performance analysis.

    Args:
        time_window: Time window for memory retrieval (e.g., "24h", "7d", "30m")
        model_name: Optional override for helper model (for testing different models)
        temperature: Model temperature for deterministic testing (default: 0.0)

    Returns:
        JSON string containing consolidation results or detailed failure analysis
    """
    correlation_id = generate_correlation_id("consolidate_shortterm")
    set_correlation_id(correlation_id)

    logger = get_logger("tools.consolidate_shortterm")
    logger.info(
        "Consolidate short-term memories request received",
        time_window=time_window,
        model_name=model_name,
        temperature=temperature,
        correlation_id=correlation_id,
    )

    try:
        # Call the consolidation service
        result = await consolidation_service.consolidate_shortterm_memories(
            time_window=time_window,
            model_name=model_name,
            temperature=temperature,
        )

        # Add tool-level metadata
        result["tool_metadata"] = {
            "tool_name": "consolidate_shortterm",
            "tool_version": "2.0",
            "systematic_evaluation": True,
            "deterministic_testing": temperature == 0.0,
        }

        logger.info(
            "Consolidate short-term memories completed",
            success=result.get("success", False),
            model_validation_success=result.get("metadata", {})
            .get("model_evaluation", {})
            .get("validation_success", False),
            correlation_id=correlation_id,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(
            "Consolidate short-term memories failed",
            error=str(e),
            error_type=type(e).__name__,
            correlation_id=correlation_id,
        )

        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "tool_metadata": {
                "tool_name": "consolidate_shortterm",
                "tool_version": "2.0",
                "error_location": "tool_level",
            },
            "correlation_id": correlation_id,
        }

        return json.dumps(error_response, indent=2)


def register_consolidate_shortterm_tools(mcp: FastMCP) -> None:
    """Register consolidate_shortterm tools with the MCP server."""
    logger = get_logger("tools.consolidate_shortterm")

    # Register the consolidate_shortterm tool
    mcp.tool(consolidate_shortterm)

    logger.debug("consolidate_shortterm tools registered")

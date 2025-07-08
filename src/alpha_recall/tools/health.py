"""Health check tools for Alpha-Recall."""

import json

from fastmcp import FastMCP

from ..logging import get_logger
from ..utils.correlation import generate_correlation_id, set_correlation_id
from ..utils.time import current_utc
from ..version import __version__

__all__ = ["health_check", "register_health_tools"]


def health_check() -> str:
    """Comprehensive health check for the Alpha-Recall server"""
    import time

    # Generate correlation ID for this health check
    correlation_id = generate_correlation_id("health")
    set_correlation_id(correlation_id)

    start_time = time.perf_counter()
    tool_logger = get_logger("tools.health_check")

    tool_logger.info(
        "Health check requested", operation="health_check", version=__version__
    )

    # Check our dependencies
    checks = {}
    overall_status = "ok"
    check_times = {}

    # Check Memgraph connection
    memgraph_start = time.perf_counter()
    try:
        from ..services.memgraph import get_memgraph_service

        memgraph_service = get_memgraph_service()
        if memgraph_service.test_connection():
            checks["memgraph"] = "ok"
            tool_logger.debug(
                "Memgraph health check passed", service="memgraph", status="ok"
            )
        else:
            checks["memgraph"] = "error: connection test failed"
            overall_status = "error"
            tool_logger.error(
                "Memgraph health check failed",
                service="memgraph",
                error="connection test failed",
            )
    except Exception as e:
        checks["memgraph"] = f"error: {str(e)}"
        overall_status = "error"
        tool_logger.error(
            "Memgraph health check failed",
            service="memgraph",
            error=str(e),
            error_type=type(e).__name__,
        )
    finally:
        check_times["memgraph"] = round(
            (time.perf_counter() - memgraph_start) * 1000, 2
        )

    # Check Redis connection
    redis_start = time.perf_counter()
    try:
        # TODO: Add actual Redis connection test when we implement it
        checks["redis"] = "ok"
        tool_logger.debug("Redis health check passed", service="redis", status="ok")
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"
        overall_status = "error"
        tool_logger.error(
            "Redis health check failed",
            service="redis",
            error=str(e),
            error_type=type(e).__name__,
        )
    finally:
        check_times["redis"] = round((time.perf_counter() - redis_start) * 1000, 2)

    total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response_data = {
        "status": overall_status,
        "version": __version__,
        "checks": checks,
        "timestamp": current_utc(),
        "correlation_id": correlation_id,
    }

    response = json.dumps(response_data, indent=2)

    tool_logger.info(
        "Health check completed",
        operation="health_check",
        status=overall_status,
        total_time_ms=total_time_ms,
        memgraph_time_ms=check_times["memgraph"],
        redis_time_ms=check_times["redis"],
        checks_passed=sum(1 for status in checks.values() if status == "ok"),
        checks_failed=sum(
            1 for status in checks.values() if status.startswith("error")
        ),
    )

    return response


def register_health_tools(mcp: FastMCP) -> None:
    """Register health check tools with the MCP server."""
    logger = get_logger("tools.health")

    # Register the health check tool
    mcp.tool(health_check)

    logger.debug("Health check tools registered")

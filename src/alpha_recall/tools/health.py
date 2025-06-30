"""Health check tools for Alpha-Recall."""

import json
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP
from ..logging import get_logger
from ..version import __version__
from ..config import settings

__all__ = ["health_check", "register_health_tools"]


def register_health_tools(mcp: FastMCP) -> None:
    """Register health check tools with the MCP server."""
    logger = get_logger("tools.health")
    
    @mcp.tool()
    def health_check() -> str:
        """Comprehensive health check for the Alpha-Recall server"""
        tool_logger = get_logger("tools.health_check")
        tool_logger.info("Health check requested")
        
        # Check our dependencies
        checks = {}
        overall_status = "ok"
        
        # Check Memgraph connection
        try:
            # TODO: Add actual Memgraph connection test when we implement it
            checks["memgraph"] = "ok"
        except Exception as e:
            checks["memgraph"] = f"error: {str(e)}"
            overall_status = "error"
        
        # Check Redis connection  
        try:
            # TODO: Add actual Redis connection test when we implement it
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"error: {str(e)}"
            overall_status = "error"
        
        response_data = {
            "status": overall_status,
            "version": __version__,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        response = json.dumps(response_data, indent=2)
        tool_logger.debug("Health check completed", status=overall_status, checks=checks)
        return response
    
    logger.debug("Health check tools registered")

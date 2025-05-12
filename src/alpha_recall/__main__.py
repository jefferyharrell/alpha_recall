"""
Main entry point for the alpha_recall MCP server.
"""

import asyncio
import sys
import json

from alpha_recall.server import mcp


def main():
    """Run the MCP server as a standalone script."""
    try:
        # This is the standard way to run an MCP server for testing
        # In production, the server will be started by the client (Claude Desktop, BoltAI, etc.)
        asyncio.run(mcp.run())
    except KeyboardInterrupt:
        print("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

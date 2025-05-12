#!/usr/bin/env python
"""
Alpha-Recall MCP Server Entry Point

This script serves as the main entry point for the Alpha-Recall MCP server.
It sets up the Python path correctly and starts the server.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
ROOT_DIR = Path(__file__).parent.absolute()
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Import packages here but don't use them yet to avoid asyncio issues


def main():
    """Run the Alpha-Recall MCP server."""
    try:
        # Import and run the FastMCP server directly
        # This avoids the "Already running asyncio in this thread" error
        from alpha_recall.server import mcp
        
        # Use asyncio.run() only once at the top level
        asyncio.run(mcp.run())
    except KeyboardInterrupt:
        print("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the server
    main()

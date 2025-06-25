#!/usr/bin/env python3
"""
Start the FastMCP server for alpha-recall
"""
import os
import sys

# Set up environment for FastMCP server (using high port to avoid conflicts)
os.environ["PYTHONPATH"] = "src"
os.environ["MCP_TRANSPORT"] = "streamable-http"
os.environ["FASTMCP_HOST"] = "localhost"
os.environ["FASTMCP_PORT"] = "9005"  # High port to avoid existing services

# Import and run
sys.path.insert(0, "src")
from alpha_recall.fastmcp_server import main

if __name__ == "__main__":
    print("🚀 Starting alpha-recall FastMCP server on localhost:9005")
    print("   Transport: streamable-http")
    print("   Use Ctrl+C to stop")
    print()
    main()
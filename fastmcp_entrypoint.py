#!/usr/bin/env python3
"""
Docker entrypoint for FastMCP server.

This ensures proper Python path setup when running in Docker.
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, '/app/src')

# Set required environment variables if not set
if not os.environ.get('PYTHONPATH'):
    os.environ['PYTHONPATH'] = '/app/src'

# Import and run the FastMCP server
from alpha_recall.fastmcp_server import main

if __name__ == "__main__":
    print("🚀 Starting FastMCP server from Docker entrypoint")
    main()
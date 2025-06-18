#!/usr/bin/env python3
"""Simple wrapper to run the emotional embeddings migration."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from alpha_recall.migrate_emotional_embeddings import main

if __name__ == "__main__":
    asyncio.run(main())
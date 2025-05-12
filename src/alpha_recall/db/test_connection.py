"""
Utility script to test Neo4j database connection.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_dir))

from src.alpha_recall.db.factory import create_db_instance
from src.alpha_recall.logging_utils import configure_logging


async def test_connection():
    """
    Test connection to the configured graph database.
    """
    # Configure logging
    logger = configure_logging()
    logger.info("Testing database connection...")
    
    try:
        # Create and connect to database
        db = await create_db_instance()
        logger.info("Connection successful!")
        
        # Test a simple query
        result = await db.execute_query("RETURN 'Connection test successful' AS message")
        logger.info(f"Query result: {result[0]['message']}")
        
        # Close connection
        await db.close()
        logger.info("Connection closed")
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)

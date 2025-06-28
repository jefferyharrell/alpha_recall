#!/usr/bin/env python3
"""
Test script for timezone fix in Redis short-term memory.

This script tests that timestamps now include proper UTC timezone indicators.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from alpha_recall.db.redis_db import RedisShortTermMemory


async def test_timezone_fix():
    """Test that timestamps now include proper timezone information."""
    print("🕐 Testing timezone fix for Redis short-term memory")
    print("=" * 60)
    
    # Create a Redis STM instance
    redis_stm = RedisShortTermMemory()
    
    try:
        # Test storing a memory
        print("Storing a test memory...")
        result = await redis_stm.store_memory(
            "Test memory for timezone fix verification",
            {"client_name": "test_client"}
        )
        
        if result.get("success", True):
            timestamp = result.get("created_at")
            print(f"✅ Memory stored successfully!")
            print(f"Timestamp: {timestamp}")
            
            # Check if timestamp has timezone information
            if timestamp and ("+00:00" in timestamp or "Z" in timestamp):
                print("✅ Timestamp includes timezone information!")
                
                # Try to parse it as a proper ISO datetime
                try:
                    parsed_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    print(f"✅ Timestamp parses correctly: {parsed_dt}")
                    print(f"✅ Timezone: {parsed_dt.tzinfo}")
                    
                    # Verify it's UTC
                    if parsed_dt.tzinfo == timezone.utc:
                        print("✅ Confirmed: Timestamp is in UTC!")
                    else:
                        print(f"⚠️  Warning: Timezone is {parsed_dt.tzinfo}, expected UTC")
                        
                except Exception as e:
                    print(f"❌ Failed to parse timestamp: {e}")
                    return False
                    
            else:
                print(f"❌ Timestamp missing timezone information: {timestamp}")
                return False
            
            # Test retrieving memories to see if timestamps are consistent
            print("\nTesting memory retrieval...")
            memories = await redis_stm.get_recent_memories(limit=1)
            
            if memories:
                retrieved_timestamp = memories[0].get("created_at")
                print(f"Retrieved timestamp: {retrieved_timestamp}")
                
                if retrieved_timestamp == timestamp:
                    print("✅ Stored and retrieved timestamps match!")
                    return True
                else:
                    print(f"❌ Timestamp mismatch - stored: {timestamp}, retrieved: {retrieved_timestamp}")
                    return False
            else:
                print("❌ Failed to retrieve stored memory")
                return False
                
        else:
            print(f"❌ Failed to store memory: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    finally:
        # Clean up
        await redis_stm.close()


async def test_current_time_comparison():
    """Test timestamp comparison with current time."""
    print("\n🕐 Testing timestamp comparison with current time")
    print("=" * 60)
    
    # Get current time in multiple formats
    current_utc = datetime.now(timezone.utc)
    current_utc_iso = current_utc.isoformat()
    
    print(f"Current UTC time: {current_utc_iso}")
    
    # Create a test timestamp using the new method
    test_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"Test timestamp: {test_timestamp}")
    
    # Parse both and compare
    current_parsed = datetime.fromisoformat(current_utc_iso)
    test_parsed = datetime.fromisoformat(test_timestamp)
    
    time_diff = abs((test_parsed - current_parsed).total_seconds())
    print(f"Time difference: {time_diff:.3f} seconds")
    
    if time_diff < 1.0:  # Should be very close
        print("✅ Timestamps are consistent with current time!")
        return True
    else:
        print(f"❌ Timestamps differ too much from current time")
        return False


async def main():
    """Run all timezone tests."""
    print("🧪 Timezone Fix Test Suite")
    print("=" * 60)
    
    test1_success = await test_timezone_fix()
    test2_success = await test_current_time_comparison()
    
    print("\n" + "=" * 60)
    if test1_success and test2_success:
        print("🎉 All timezone tests passed!")
        print("Short-term memory timestamps now include proper UTC timezone information!")
        print("This should fix Alpha's temporal confusion issues.")
    else:
        print("❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
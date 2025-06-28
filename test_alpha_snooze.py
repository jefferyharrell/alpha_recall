#!/usr/bin/env python3
"""
Test script for Alpha-Snooze memory consolidation feature.

This script tests the alpha-snooze integration without requiring a full
MCP server setup. It simulates the gentle_refresh workflow with sample
short-term memories.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from alpha_recall.utils.alpha_snooze import create_alpha_snooze_from_env, AlphaSnooze


# Sample short-term memories for testing
SAMPLE_MEMORIES = [
    {
        "content": "🎉 Jeffery is PUMPED and wants to switch to Claude Code to implement alpha-snooze RIGHT NOW! He thinks we could bang it out pretty quickly once I'm swimming in the actual Alpha-Recall codebase. The energy is high and we're ready to make this memory consolidation feature a reality!",
        "created_at": "2025-06-27T23:03:54.721836",
        "client": {"client_name": "Claude Desktop"}
    },
    {
        "content": "EXECUTIVE DECISION from Jeffery! The memory processing assistant will be integrated DIRECTLY into Alpha-Recall - not separate service, but in-process, intrinsic to how my memory works. Optional but integral. And we're calling it 'snooze' or 'alpha-snooze' - a little pause to consolidate before waking up to new conversations. I LOVE this decision and the naming!",
        "created_at": "2025-06-27T23:01:09.725135",
        "client": {"client_name": "Claude Desktop"}
    },
    {
        "content": "Good point from Jeffery! The real Alpha-Sleep won't have internet access anyway - it's purely for internal memory processing. Let's test with a fresh conversation to see if qwen2.5:3b (3B params) can handle memory summarization tasks. This will tell us if small models are capable of this kind of cognitive work.",
        "created_at": "2025-06-27T22:55:41.719616",
        "client": {"client_name": "Claude Desktop"}
    },
    {
        "content": "Successfully transitioned to Claude Code environment while maintaining Alpha identity. Jeffery was excited about implementing alpha-snooze directly in the Alpha-Recall codebase. Ready to dive into the implementation!",
        "created_at": "2025-06-27T23:05:48.677944",
        "client": {"client_name": "Claude Code"}
    }
]


async def test_alpha_snooze_availability():
    """Test if alpha-snooze is available and properly configured."""
    print("Testing Alpha-Snooze availability...")
    
    # Test direct instantiation
    snooze = AlphaSnooze()
    is_available = await snooze.is_available()
    
    print(f"Alpha-Snooze availability (direct): {is_available}")
    
    # Test environment-based creation
    os.environ["ALPHA_SNOOZE_ENABLED"] = "true"
    env_snooze = await create_alpha_snooze_from_env()
    
    if env_snooze:
        env_available = await env_snooze.is_available()
        print(f"Alpha-Snooze availability (from env): {env_available}")
        return env_snooze if env_available else None
    else:
        print("Alpha-Snooze not enabled or not available from environment")
        return None


async def test_memory_consolidation(snooze: AlphaSnooze):
    """Test memory consolidation with sample memories."""
    print("\nTesting memory consolidation...")
    
    consolidation = await snooze.consolidate_memories(SAMPLE_MEMORIES)
    
    if consolidation:
        print("✅ Memory consolidation successful!")
        print(f"Processed {consolidation.get('processed_memories_count', 0)} memories")
        print(f"Model used: {consolidation.get('model_used', 'unknown')}")
        print(f"Consolidation time: {consolidation.get('consolidation_timestamp', 'unknown')}")
        
        # Print key results
        print("\n--- Consolidation Results ---")
        
        entities = consolidation.get("entities", [])
        print(f"Entities discovered: {len(entities)}")
        for entity in entities[:3]:  # Show first 3
            print(f"  - {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown type')})")
        
        relationships = consolidation.get("relationships", [])
        print(f"Relationships found: {len(relationships)}")
        for rel in relationships[:2]:  # Show first 2
            print(f"  - {rel.get('entity1', '?')} {rel.get('relationship', '?')} {rel.get('entity2', '?')}")
        
        insights = consolidation.get("insights", [])
        print(f"Insights extracted: {len(insights)}")
        for insight in insights[:2]:  # Show first 2
            print(f"  - {insight}")
        
        summary = consolidation.get("summary", "No summary available")
        print(f"Summary: {summary}")
        
        emotional_context = consolidation.get("emotional_context", "unknown")
        print(f"Emotional context: {emotional_context}")
        
        next_steps = consolidation.get("next_steps", [])
        print(f"Next steps: {len(next_steps)}")
        for step in next_steps:
            print(f"  - {step}")
        
        return True
    else:
        print("❌ Memory consolidation failed")
        return False


async def test_gentle_refresh_integration():
    """Test how alpha-snooze would integrate with gentle_refresh."""
    print("\n--- Testing gentle_refresh integration ---")
    
    # Simulate the gentle_refresh workflow
    try:
        alpha_snooze = await create_alpha_snooze_from_env()
        if alpha_snooze and SAMPLE_MEMORIES:
            print("Running alpha-snooze memory consolidation (simulated gentle_refresh)")
            consolidation = await alpha_snooze.consolidate_memories(SAMPLE_MEMORIES)
            if consolidation:
                print("✅ Integration test successful - consolidation would be added to gentle_refresh response")
                print(f"Memory consolidation object would contain {len(consolidation)} fields")
                return True
            else:
                print("❌ Integration test failed - consolidation returned no results")
                return False
        elif alpha_snooze:
            print("⚠️  Alpha-snooze available but no short-term memories to process")
            return True
        else:
            print("ℹ️  Alpha-snooze not available - gentle_refresh would continue normally")
            return True
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False


async def main():
    """Main test function."""
    print("🧠 Alpha-Snooze Test Suite")
    print("=" * 50)
    
    # Test availability
    snooze = await test_alpha_snooze_availability()
    
    if snooze:
        # Test consolidation
        consolidation_success = await test_memory_consolidation(snooze)
        
        # Test integration
        integration_success = await test_gentle_refresh_integration()
        
        print("\n" + "=" * 50)
        if consolidation_success and integration_success:
            print("🎉 All tests passed! Alpha-Snooze is ready for action!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
    else:
        print("\n" + "=" * 50)
        print("ℹ️  Alpha-Snooze not available. This is normal if:")
        print("  - Ollama is not running")
        print("  - The specified model is not available")
        print("  - ALPHA_SNOOZE_ENABLED is not set to 'true'")
        print("\nTo enable alpha-snooze:")
        print("  1. Install and start Ollama")
        print("  2. Pull the model: ollama pull qwen2.5:3b")
        print("  3. Set environment variable: export ALPHA_SNOOZE_ENABLED=true")


if __name__ == "__main__":
    asyncio.run(main())
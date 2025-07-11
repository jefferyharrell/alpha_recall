#!/usr/bin/env python3
"""
Consolidation stress tester for Alpha-Recall.

Tests the consolidate_shortterm tool repeatedly to identify failure patterns.
"""

import asyncio
import json
from datetime import datetime

from fastmcp import Client
from rich import print


async def test_consolidation(
    test_num: int,
    server_url: str = "http://localhost:9005/mcp/",
    model_name: str | None = None,
):
    """Run a single consolidation test."""
    try:
        async with Client(server_url) as client:
            params = {"time_window": "24h"}
            if model_name:
                params["model_name"] = model_name
            result = await client.call_tool("consolidate_shortterm", params)
            content = result.content[0].text if result.content else ""

            # Parse and display the result
            data = json.loads(content)

            print(f"\n🧪 Test #{test_num} - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)

            if data.get("success"):
                metadata = data.get("metadata", {})
                print("✅ SUCCESS")
                print(f"Model: {metadata.get('model_name', 'unknown')}")
                print(f"Memories: {metadata.get('input_memories_count', 0)}")
                print(f"Time: {metadata.get('processing_time_ms', 0)}ms")
                print(f"Approach: {metadata.get('approach', 'unknown')}")
                print(f"Temperature: {metadata.get('temperature', 'unknown')}")

                if "narrative" in data:
                    print("\nNarrative:")
                    print(data["narrative"])
                else:
                    print(f"\nRaw result: {content}")
            else:
                print(f"❌ FAILED: {data.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n🧪 Test #{test_num} - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        print(f"💥 EXCEPTION: {e}")


async def run_stress_test(
    num_tests: int = 10, delay: float = 2.0, model_name: str | None = None
):
    """Run multiple consolidation tests."""
    model_info = f" using {model_name}" if model_name else ""
    print(
        f"🧪 Starting {num_tests} consolidation tests{model_info} with {delay}s delay..."
    )

    for i in range(1, num_tests + 1):
        await test_consolidation(i, model_name=model_name)
        if i < num_tests:
            await asyncio.sleep(delay)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Alpha-Recall consolidation")
    parser.add_argument(
        "--tests", "-n", type=int, default=10, help="Number of tests to run"
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=2.0, help="Delay between tests (seconds)"
    )
    parser.add_argument(
        "--model", "-m", type=str, help="Override model (e.g., granite3.3:2b)"
    )

    args = parser.parse_args()

    await run_stress_test(args.tests, args.delay, args.model)


if __name__ == "__main__":
    asyncio.run(main())

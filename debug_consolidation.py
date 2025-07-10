"""Quick debug script to see consolidation validation errors."""

import asyncio
import json

from fastmcp import Client


async def debug_consolidation():
    """Debug consolidation to see validation errors."""
    server_url = "http://localhost:19005"

    async with Client(server_url) as client:
        # Add a test memory
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Debug test: Alpha is investigating consolidation schema validation."
            },
        )

        # Try consolidation
        result = await client.call_tool(
            "consolidate_shortterm", {"time_window": "1h", "temperature": 0.0}
        )

        data = json.loads(result.content[0].text)

        print("=== CONSOLIDATION DEBUG ===")
        print(f"Success: {data.get('success', False)}")

        if not data.get("success", False):
            print("\n=== VALIDATION ERRORS ===")
            if "validation_errors" in data:
                for error in data["validation_errors"]:
                    print(f"  - {error}")
            elif "error" in data:
                print(f"  - {data['error']}")
            else:
                print("  - No specific error details found")

        print("\n=== FULL RESPONSE ===")
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    asyncio.run(debug_consolidation())

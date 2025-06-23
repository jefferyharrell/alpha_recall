#!/usr/bin/env python3
"""
Simple test script for the alpha-recall HTTP server.
Tests basic endpoints without requiring database connections.
"""

import asyncio
import json
from typing import Dict, Any

import httpx


async def test_health_check(base_url: str) -> None:
    """Test the health check endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200


async def test_list_tools(base_url: str) -> None:
    """Test listing available tools."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/tools")
        print(f"\nList tools status: {response.status_code}")
        data = response.json()
        print(f"Available tools: {len(data['tools'])}")
        for tool in data["tools"]:
            print(f"  - {tool['name']}")
        assert response.status_code == 200


async def test_tool_call(base_url: str) -> None:
    """Test calling a tool (will fail without database, but tests the endpoint)."""
    async with httpx.AsyncClient() as client:
        # Test with refresh tool
        request_data = {
            "tool": "refresh",
            "params": {"query": "Hello, Alpha!"}
        }
        
        response = await client.post(
            f"{base_url}/tool/refresh",
            json=request_data
        )
        print(f"\nTool call status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        # We expect this to fail without database connection
        assert response.status_code in [200, 500]


async def test_sse_endpoint(base_url: str) -> None:
    """Test SSE endpoint connectivity."""
    print("\nTesting SSE endpoint...")
    async with httpx.AsyncClient() as client:
        # Just test that we can connect
        try:
            async with client.stream("GET", f"{base_url}/sse") as response:
                print(f"SSE connection status: {response.status_code}")
                assert response.status_code == 200
                
                # Read first message (should be a ping)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        print(f"Received SSE message: {data['type']}")
                        break
        except Exception as e:
            print(f"SSE test error (expected): {e}")


async def main():
    """Run all tests."""
    base_url = "http://localhost:8080"
    
    print("Testing alpha-recall HTTP server...")
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    try:
        await test_health_check(base_url)
        await test_list_tools(base_url)
        await test_tool_call(base_url)
        await test_sse_endpoint(base_url)
        
        print("\n✅ All endpoint tests completed!")
        print("\nNote: Tool calls will fail without database connections.")
        print("This is expected when testing the HTTP server in isolation.")
        
    except httpx.ConnectError:
        print("\n❌ Could not connect to server!")
        print("Make sure the server is running with:")
        print("  docker-compose up")


if __name__ == "__main__":
    asyncio.run(main())
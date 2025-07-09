"""Shared fixtures for Alpha-Recall E2E tests."""

import asyncio
import subprocess
import time
from pathlib import Path

import pytest
from fastmcp import Client


@pytest.fixture(scope="session")
def test_stack():
    """Spin up the full Alpha-Recall test stack."""
    # Path to our test compose file
    compose_file = Path(__file__).parent.parent / "docker" / "e2e.yml"
    project_name = "alpha-recall-e2e-test"

    try:
        # Start the test stack
        print(f"Starting test stack with compose file: {compose_file}")
        print(f"Project name: {project_name}")

        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "-p",
                project_name,
                "up",
                "-d",
                "--build",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        print(f"Docker compose stdout: {result.stdout}")
        if result.stderr:
            print(f"Docker compose stderr: {result.stderr}")

        # Wait for the server to be ready - test actual MCP interface
        server_url = "http://localhost:19006/mcp/"
        max_attempts = 90  # Increased timeout for model downloads (3 minutes)

        async def check_server():
            async with Client(server_url) as client:
                await client.ping()

        for attempt in range(max_attempts):
            try:
                # Try to initialize the MCP server using proper MCP client
                asyncio.run(check_server())
                print(f"Server ready after {attempt + 1} attempts")
                break
            except Exception as e:
                # Log the specific error for debugging
                if attempt % 15 == 0:  # Log every 30 seconds
                    print(
                        f"Attempt {attempt + 1}/{max_attempts} - server not ready: {e}"
                    )

            if attempt == max_attempts - 1:
                # Show container logs on failure for debugging
                try:
                    logs_result = subprocess.run(
                        ["docker", "logs", "alpha-recall-test-server"],
                        capture_output=True,
                        text=True,
                    )
                    print(f"Container logs:\n{logs_result.stdout}")
                    if logs_result.stderr:
                        print(f"Container stderr:\n{logs_result.stderr}")
                except Exception:
                    pass
                raise RuntimeError("Test server failed to start within 180 seconds")

            if attempt % 15 == 0 and attempt > 0:  # Progress update every 30 seconds
                print(
                    f"Still waiting... attempt {attempt + 1}/{max_attempts} - allowing time for model downloads"
                )
            time.sleep(2)

        yield server_url

    finally:
        # Clean up the test stack
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "-p",
                project_name,
                "down",
                "-v",
                "--remove-orphans",
            ],
            capture_output=True,
        )

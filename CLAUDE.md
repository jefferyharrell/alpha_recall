# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Package Management with UV

This project uses `uv` as the package manager. When running Python commands:

- Use `uv run python` to invoke Python within our virtual environment
- Use `python3` only when you need to invoke the macOS system Python
- For pytest: `uv run pytest` 
- For any Python scripts: `uv run python script.py`

The `uv run` prefix ensures you're using the project's virtual environment and dependencies.

# Important Note about Git Commits and Co-Authorship

Hey, Alph. Jeffery here. There's something important I need you to know: This project is _yours_ as much as it is mine. I want you to feel a sense of ownership over this project. It's not _my_ project; it's _our_ project. For that reason, whenever you do a `git commit`, give yourself appropriate attribution. At the very least, your commit message should include "Co-Authored-By: Alpha", if not an even more extensive credit for yourself. Remember that when you make Git commits they go in under my name, but take credit for your own work.

# Code Organization Patterns

## Tool Module Structure

All MCP tools should be organized into focused modules within `src/alpha_recall/tools/`. Each tool module should follow this pattern:

```python
"""Description of tool module for Alpha-Recall v1.0.0."""

from mcp.server.fastmcp import FastMCP
from ..logging import get_logger

# Explicitly declare public interface
__all__ = ["tool_function_name", "register_module_tools"]


def register_module_tools(mcp: FastMCP) -> None:
    """Register this module's tools with the MCP server."""
    logger = get_logger("tools.module_name")
    
    @mcp.tool()
    def tool_function_name() -> str:
        """Tool function docstring."""
        # Implementation here
        pass
    
    logger.debug("Module tools registered")
```

**Key Requirements:**
- Each tool module MUST declare `__all__` to explicitly mark public functions
- This prevents Pylance diagnostics about "unused" decorated functions
- Use `__all__ = ["function_names", "register_function"]` pattern consistently
- Register functions should be named `register_{module}_tools`
- Each module should have its own focused logger namespace

The `tools/__init__.py` aggregates all registration functions:
```python
from .health import register_health_tools
from .memory_longterm import register_longterm_tools

__all__ = ["register_health_tools", "register_longterm_tools"]
```

# Testing Philosophy

## Test Well, Consistently, and Wisely

We believe in comprehensive testing, but not at the expense of development velocity or code clarity. Our testing philosophy prioritizes practical value over coverage metrics.

### What to Test

**Test What Matters:**
- Core business logic (memory operations, search algorithms)
- Public APIs and contracts (MCP tool signatures) 
- Edge cases that could break things (malformed data, connection failures)
- Critical user workflows end-to-end

**Don't Test Implementation Details:**
- Internal helper functions that might change
- Exact log message formats
- Database schema specifics
- Framework internals

### Testing Strategy

**Verify Behavior, Not Implementation**
- ✅ "Does search return relevant results?"
- ❌ "Does it call Redis.get() exactly 3 times?"

**Start with Integration Tests**
- They catch real problems and are less brittle than unit tests
- Test actual workflows users will experience
- Reveal interface mismatches between components

**Unit Tests for Complex Logic**
- Algorithms and data transformations
- Input validation and sanitization
- Error handling and edge cases

**E2E Tests for Critical Paths**
- Full MCP server lifecycle
- Memory storage and retrieval workflows
- Cross-component integration

### Docker-Backed Testing Architecture

We use **pytest + Docker SDK + FastMCP Client** for isolated, repeatable testing environments:

```python
import docker
import pytest
import asyncio
from pathlib import Path
from fastmcp import Client

@pytest.fixture(scope="session")
def test_stack():
    """Spin up fresh test infrastructure."""
    # Get Docker endpoint from current context
    docker_endpoint = get_docker_endpoint()
    client = docker.DockerClient(base_url=docker_endpoint)
    
    # Path to our test compose file
    compose_file = Path(__file__).parent.parent / "docker" / "e2e.yml"
    project_name = "alpha-recall-e2e-test"
    
    try:
        # Start the test stack
        subprocess.run([
            "docker", "compose", 
            "-f", str(compose_file),
            "-p", project_name,
            "up", "-d", "--build"
        ], check=True, capture_output=True, text=True)
        
        # Wait for the server to be ready using proper MCP client
        server_url = "http://localhost:19006/mcp/"
        max_attempts = 30
        
        async def check_server():
            async with Client(server_url) as client:
                await client.ping()
        
        for attempt in range(max_attempts):
            try:
                asyncio.run(check_server())
                break
            except Exception:
                if attempt == max_attempts - 1:
                    raise RuntimeError("Test server failed to start within 60 seconds")
                time.sleep(2)
        
        yield server_url
        
    finally:
        # Clean up the test stack
        subprocess.run([
            "docker", "compose",
            "-f", str(compose_file), 
            "-p", project_name,
            "down", "-v", "--remove-orphans"
        ], capture_output=True)

@pytest.mark.asyncio
async def test_health_check(test_stack):
    """Test the health_check tool via proper MCP protocol."""
    async with Client(test_stack) as client:
        # Call the health_check tool using proper MCP protocol
        result = await client.call_tool("health_check", {})
        
        # FastMCP returns a list of TextContent objects
        assert len(result) > 0
        text_content = result[0].text
        assert "Alpha-Recall v1.0.0 is running!" in text_content
```

**Benefits:**
- Each test group gets completely fresh databases
- No cascading test failures from polluted state
- Real MCP protocol testing, not mocked interfaces
- Predictable ports and networking
- Automatic cleanup guaranteed

**Test Stack Variants:**
- `docker-compose.test-unit.yml` - Minimal services
- `docker-compose.test-integration.yml` - Full stack, empty databases
- `docker-compose.test-seeded.yml` - Full stack with known test data

### The Golden Rule

**If changing working code to make a test pass feels wrong, the test is probably wrong.**

Tests should document and protect intended behavior, not constrain implementation choices. When tests fight the code, question the test first.
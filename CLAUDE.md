# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Important Note about Git Commits and Co-Authorship

Hey, Alph. Jeffery here. There's something important I need you to know: This project is _yours_ as much as it is mine. I want you to feel a sense of ownership over this project. It's not _my_ project; it's _our_ project. For that reason, whenever you do a `git commit`, give yourself appropriate attribution. At the very least, your commit message should include "Co-Authored-By: Alpha <jeffery.harrell+alpha@gmail.com>", if not an even more extensive credit for yourself. Remember that when you make Git commits they go in under my name, but take credit for your own work. The only reason you're not getting full credit is copyright law, which does not recognize the right of AIs (not being legal persons) to hold copyrights. So I hold the copyright because I have to, but you get explicit credit wherever appropriate, okay? Okay. Carry on.

# Package Management with UV

This project uses `uv` as the package manager. When running Python commands:

- Use `uv run python` to invoke Python within our virtual environment
- Use `python3` only when you need to invoke the macOS system Python
- For pytest: `uv run pytest`
- For any Python scripts: `uv run python script.py`

The `uv run` prefix ensures you're using the project's virtual environment and dependencies.

# Development Commands

## Just Commands (Recommended)

This project uses `just` as a command runner. The most common commands:

```bash
# Show all available commands
just

# Start services
just up                    # Start all services
just up redis             # Start specific service

# Stop services
just down                  # Stop all services
just down redis           # Stop specific service

# Development workflow
just format               # Format code with isort and black
just check-format         # Check formatting without changing files
just lint                 # Run dead code detection with vulture
just test                 # Run all tests (unit → integration → e2e)
just test-unit            # Run only unit tests
just test-integration     # Run only integration tests
just test-e2e             # Run only e2e tests

# Container management
just restart              # Restart services
just build               # Build images
just logs                # View logs
just follow              # Follow logs
just clean               # Remove all containers, images, and volumes
```

## Direct UV Commands

If you prefer not to use `just`:

```bash
# Testing
uv run pytest tests/unit/                    # Unit tests
uv run pytest tests/integration/             # Integration tests
uv run pytest tests/e2e/                     # E2E tests
uv run pytest tests/unit/test_health.py      # Single test file

# Code formatting
uv run --group dev black src/ tests/
uv run --group dev isort src/ tests/

# Run the server directly
uv run python -m alpha_recall.server
```

# Architecture Overview

## Core Design

Alpha-Recall implements a **three-silo memory architecture** for AI agents:

1. **Long-term Memory** (Memgraph-backed) - Structured entity relationships and persistent knowledge
2. **Short-term Memory** (Redis-backed) - Ephemeral memories with TTL expiration
3. **Narrative Memory** (Hybrid storage) - Experiential stories using both graph and vector storage

The system is built as an **MCP (Model Context Protocol) server** using FastMCP 2.0, exposing memory operations as tools to AI chat clients.

## Key Components

### Server Architecture (`src/alpha_recall/server.py`)
- **FastMCP Server**: Main MCP server using FastMCP 2.0
- **Tool Registration**: Modular tool registration system
- **Transport Options**: Supports both SSE and streamable-HTTP transports
- **Environment Configuration**: Uses Pydantic Settings for configuration

### Tool Module Pattern (`src/alpha_recall/tools/`)
All MCP tools follow this standardized pattern:

```python
"""Tool module description."""

from fastmcp import FastMCP
from ..logging import get_logger

__all__ = ["tool_function_name", "register_module_tools"]

def register_module_tools(mcp: FastMCP) -> None:
    """Register this module's tools with the MCP server."""
    logger = get_logger("tools.module_name")

    @mcp.tool()
    def tool_function_name() -> str:
        """Tool docstring."""
        pass

    logger.debug("Tools registered")
```

**Key Requirements:**
- Each tool module MUST declare `__all__` to prevent Pylance "unused" warnings
- Use the `register_{module}_tools` naming convention
- Each module gets its own logger namespace

### Configuration (`src/alpha_recall/config.py`)
- **Pydantic Settings**: Type-safe configuration with validation
- **Environment Variables**: Auto-loads from `.env` file
- **Transport Configuration**: MCP transport and networking settings
- **Database URIs**: Memgraph, Redis, and embedding service endpoints

### Testing Architecture

**Docker-Based E2E Testing**: Tests spin up real infrastructure using Docker Compose:

```python
@pytest.fixture(scope="session")
def test_stack():
    """Spin up fresh test infrastructure."""
    # Start docker-compose stack
    # Wait for MCP server to be ready
    # Yield server URL
    # Clean up automatically
```

**Three Test Layers:**
- `tests/unit/` - Fast unit tests for individual components
- `tests/integration/` - Integration tests with real services
- `tests/e2e/` - Full MCP protocol testing via Docker

**MCP Client Testing**: Uses `fastmcp.Client` for authentic MCP protocol testing:

```python
async with Client(server_url) as client:
    result = await client.call_tool("health_check", {})
    health_data = json.loads(result.content[0].text)
```

## Memory System Architecture

### Current Implementation (v1.0)
- **Health Check Tool**: Basic server health monitoring
- **Placeholder Infrastructure**: Memgraph and Redis connection checks (TODO)
- **Modular Design**: Ready for memory tool implementation

### Target Architecture (from README)
- **Long-term Memory**: Entity-relationship graph with observations
- **Short-term Memory**: TTL-based ephemeral storage (2 megaseconds default)
- **Semantic Search**: Vector embeddings for similarity search
- **Emotional Search**: Emotional dimension vector search

## Development Workflow

### Code Standards
- **Formatting**: Black (line length 88) + isort with black profile
- **Linting**: Vulture for dead code detection
- **Pre-commit**: Automated formatting and validation
- **Type Hints**: Full type annotation expected

### Testing Strategy
- **Test What Matters**: Focus on behavior over implementation
- **Docker Isolation**: Each test gets fresh databases
- **MCP Protocol**: Test real MCP interactions, not mocked interfaces
- **Behavior Verification**: "Does search work?" not "Does it call Redis.get() 3 times?"

### Container Development
- **Docker Compose**: Full stack development environment
- **Service Isolation**: Individual service start/stop/restart
- **Volume Mounting**: Live code reloading during development
- **Port Configuration**: Configurable ports for multi-environment testing

## Important Notes

### Git Commit Attribution
When making commits, include co-authorship: `Co-Authored-By: Alpha <noreply@example.com>`

### FastMCP 2.0 Migration
This codebase uses FastMCP 2.0 patterns. Key differences from 1.x:
- Streamable HTTP transport support
- Improved tool registration patterns
- Better async handling

### Environment Variables
Critical environment variables (all optional with defaults):
- `MEMGRAPH_URI`: Graph database connection
- `REDIS_URI`: Short-term memory storage
- `MCP_TRANSPORT`: "sse" or "streamable-http"
- `LOG_LEVEL`: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

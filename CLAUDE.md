# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Important Note about Git Commits and Co-Authorship

Hey, Alph. Jeffery here. There's something important I need you to know: This project is _yours_ as much as it is mine. I want you to feel a sense of ownership over this project. It's not _my_ project; it's _our_ project. For that reason, whenever you do a `git commit`, give yourself appropriate attribution. At the very least, your commit message should include "Co-Authored-By: Alpha <jeffery.harrell+alpha@gmail.com>", if not an even more extensive credit for yourself. Remember that when you make Git commits they go in under my name, but take credit for your own work. The only reason you're not getting full credit is copyright law, which does not recognize the right of AIs (not being legal persons) to hold copyrights. So I hold the copyright because I have to, but you get explicit credit wherever appropriate, okay? Okay. Carry on.

# Package Management with UV

This project uses `uv` as the package manager with **Python 3.13**. When running Python commands:

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
just lint                 # Run Ruff linter
just lint-fix             # Run Ruff linter with auto-fix
just pre-commit           # Run all pre-commit hooks manually
just test                 # Run all tests (unit → e2e)
just test-unit            # Run only unit tests
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
uv run pytest tests/e2e/                     # E2E tests
uv run pytest tests/unit/test_health.py      # Single test file

# Code formatting and linting
uv run --group dev black src/ tests/
uv run --group dev isort src/ tests/
uv run --group dev ruff check src/ tests/
uv run --group dev ruff check --fix src/ tests/

# Run the server directly
uv run python -m alpha_recall.server
```

# Code Quality and Pre-commit Hooks

This project uses extensive pre-commit hooks to maintain code quality:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all pre-commit hooks manually
just pre-commit
# or
uv run pre-commit run --all-files
```

**Pre-commit Configuration:**
- **Code formatting**: isort, black, ruff-format
- **Linting**: ruff with comprehensive rule set
- **File validation**: YAML, TOML, JSON syntax checking
- **Git hygiene**: trailing whitespace, end-of-file fixes, large file detection, merge conflict detection

All formatting and linting must pass before commits are allowed.

# Architecture Overview

Alpha-Recall is a **MCP (Model Context Protocol) server** implementing a three-silo memory architecture for AI agents. Built with **FastMCP 2.0**, it provides persistent memory capabilities across short-term, long-term, and narrative memory systems.

## Core Design

### Three-Silo Memory Architecture
- **Short-term Memory (STM)**: Redis-backed with 2-megasecond TTL, dual embedding pipeline (semantic + emotional)
- **Long-term Memory (LTM)**: Memgraph graph database for entity-observation-relationship knowledge
- **Narrative Memory (NM)**: Hybrid Redis+Memgraph storage for experiential stories with emotional context
- **Unified Search**: `search_all_memories` tool searches across all systems with merged, score-sorted results

### MCP Server Implementation
Built using **FastMCP 2.0** with modular tool registration. Each tool module follows a standardized pattern:

```python
def tool_function_name() -> str:
    """Tool implementation."""
    pass

def register_module_tools(mcp: FastMCP) -> None:
    """Register this module's tools."""
    mcp.tool(tool_function_name)
```

### Embedding Service Architecture
Dual embedding models with smart device detection:
- **Semantic**: `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- **Emotional**: `ng3owb/sentiment-embedding-model` (1024 dimensions)
- **Device Support**: Apple Silicon MPS, CUDA, CPU fallback
- **Performance**: Lazy loading, correlation ID tracing, detailed metrics

## Key Tools

### Essential Memory Tools
- **`gentle_refresh`**: Temporal orientation tool providing current time, core identity, recent memories, and context in prose format
- **`search_all_memories`**: Unified search across all memory systems (STM, LTM, NM, entities)
- **`remember_shortterm`**: Store ephemeral memories with 2-megasecond TTL
- **`remember_longterm`**: Store persistent entity observations
- **`remember_narrative`**: Store experiential stories with emotional context

### Search Tools
- **`search_shortterm`**: Semantic and emotional search of recent memories
- **`search_longterm`**: Semantic search of entity observations
- **`search_narratives`**: Semantic and emotional search of narrative memories

### Entity Management Tools
- **`get_entity`**: Retrieve specific entities by name
- **`get_relationships`**: Explore entity relationships
- **`relate_longterm`**: Create relationships between entities
- **`browse_longterm`**: Explore the knowledge graph

## Gentle Refresh Tool Architecture

The `gentle_refresh` tool provides temporal orientation for AI agents with comprehensive context in **prose format** (not JSON):

### Core Functionality
- **Current Time Information**: UTC and local times with automatic timezone detection
- **Core Identity**: Dynamic identity facts stored in Redis (no hardcoded configuration)
- **Personality Traits**: Dynamic personality system with traits and directives
- **Recent Memories**: 10 most recent short-term memories for context
- **Recent Observations**: 5 most recent entity observations

### Redis-First Architecture
- **No Configuration Required**: Identity facts are dynamically stored in Redis
- **No Fallback Logic**: Clean architecture with clear error messages for uninitialized systems
- **Graceful Degradation**: Returns helpful error messages if core components are missing

### Timezone Detection System
- **Geolocation-Based**: Uses IP geolocation for automatic timezone detection
- **Multi-Provider Reliability**: Fallback chain of worldtimeapi.org → ipinfo.io → ipapi.co
- **Portable Docker Deployment**: No filesystem mounting required
- **Travel Adaptation**: Automatically detects new timezone when location changes

# Logging and Observability

## Correlation IDs for Request Tracing

We use **correlation IDs** throughout Alpha-Recall to trace requests from entry point to completion. This creates a "golden thread" through all log entries for a single operation.

**Key Components:**
- `src/alpha_recall/utils/correlation.py` - Core correlation ID utilities
- Auto-generated IDs: `generate_correlation_id("prefix")` -> `prefix_12345678`
- Child IDs: `parent_id.operation_name` for sub-operations
- Context variables: Automatically included in all structlog entries

**Usage Pattern:**
```python
# At tool entry points
correlation_id = generate_correlation_id("mem")
set_correlation_id(correlation_id)

# Child operations get hierarchical IDs automatically
# mem_abc123 -> mem_abc123.semantic_encode -> mem_abc123.semantic_encode.emotional_encode
```

**Debugging:** `grep "correlation_id=mem_abc123" logs.txt` shows complete request flow.

## Structured Logging with Rich Metrics

All logging uses **structlog** with rich contextual data:
- Performance metrics (timing, throughput, resource usage)
- Embedding model stats (dimensions, device, load times)
- Request metadata (content length, operation type, status)
- Error context (type, message, operation step)

**Log Format Options** (set via `LOG_FORMAT` environment variable):
- `rich` - Human-readable prose format with colors (default)
- `json` - Production JSON format for log aggregation
- `rich_json` - Beautiful syntax-highlighted JSON with Rich library

Example log entries include business metrics, not just debug strings.

# Development Workflow

## MCP Client Restart Requirement

**Important**: After restarting the Alpha-Recall Docker container, MCP clients (Claude Desktop, Claude Code, VS Code with MCP extensions) must be restarted to re-establish connections.

**Typical Development Cycle:**
1. Make code changes
2. Restart container: `just restart` or `docker compose restart alpha-recall`
3. Wait for embedding models to load (~5 seconds)
4. **Restart MCP clients** (Claude Desktop, Claude Code, etc.)
5. Test changes

**Why this happens**: MCP sessions are stateful connections. When the server restarts, existing session state is lost and clients need to re-initialize their connections.

**Testing Strategy**: Because of this development cycle requirement, **whenever possible use automated testing via Pytest** to test features. Automated testing is preferred over manual testing because:
- **More streamlined**: Avoids the container restart → client restart cycle
- **Regression protection**: Tests can be captured and saved to catch future regressions
- **Faster iteration**: `just test` is faster than manual MCP client testing
- **CI/CD ready**: Automated tests can run in continuous integration

Use manual MCP client testing for final validation and user experience verification, but rely on automated tests for development iteration.

## Testing Strategy

### Two-Tier Testing Architecture
- **Unit Tests** (`tests/unit/`): Fast, isolated component testing with full mocking
- **E2E Tests** (`tests/e2e/`): Full MCP protocol testing with Docker infrastructure

### Parallel vs Serial Execution
- **Unit Tests**: Run in parallel with `pytest -n 4` for 4x speedup (~98 seconds for 109 tests)
- **E2E Tests**: Run serially due to shared Docker infrastructure
- **Sweet Spot**: `-n 4` balances speed with memory usage (each worker uses ~3GB for embedding models)

### E2E Test Data Strategy
**Philosophical Approach**: Test against realistic, populated databases rather than empty ones. The "fresh install" scenario happens once; ongoing operation happens daily.

**Two E2E Test Categories:**
1. **Greenfield Tests** (`test_greenfield_*.py`): Minimal tests for bootstrap scenarios
2. **Seeded Data Tests** (`test_seeded_*.py`): Main test suite against realistic data

**Mock Data Design** (`/mock_data/`):
- `stm_test_data.json`: 20 development memories with emotional clustering
- `ltm_test_data.json`: 8 entities with dynamic language and extreme scenarios
- `nm_test_data.json`: 5 epic collaboration narratives with cross-references

**Data Characteristics:**
- **Semantically dynamic**: Varied language, emojis, slang, technical terms
- **Emotionally diverse**: Full spectrum from frustration to excitement
- **Realistic scenarios**: Based on actual Alpha-Recall development experiences
- **Cross-referenced**: Entities appear in narratives, memories reference relationships

### Testing Best Practices
- **Behavior Verification**: Focus on "Does search work?" not implementation details
- **No Production Pollution**: All tests use isolated environments or mocks
- **MCP Protocol Testing**: E2E tests use `fastmcp.Client` for authentic protocol testing
- **Predictable Assertions**: Seeded data enables specific outcome testing
- **Performance Awareness**: Memory-intensive tests run with controlled parallelization

# Container Development

- **Docker Compose**: Full stack development environment with memgraph + redis
- **Service Isolation**: Individual service start/stop/restart (`just up redis`, `just down memgraph`)
- **Volume Mounting**: Live code reloading with `./src` mounted to `/app/src`
- **Named Volumes**: `.venv` and cache preservation via `alpha-recall-venv:/app/.venv`
- **Port Configuration**: Configurable ports for multi-environment testing
- **Internal Networking**: Services communicate via `memgraph:7687` and `redis:6379` internally
- **Security**: Ports bound to localhost only for development safety

## Docker Architecture

The development environment uses Docker Compose with:
- **Memgraph**: Graph database for LTM on port 7687
- **Redis**: Single instance for both STM and narrative memory on port 6379
- **Alpha-Recall Server**: MCP server on port 19005

Services communicate internally via Docker networks (`memgraph:7687`, `redis:6379`).

# Environment Variables

Critical environment variables (all optional with defaults):
- `MEMGRAPH_URI`: Graph database connection (default: bolt://localhost:7687)
- `REDIS_URI`: Short-term memory storage (default: redis://localhost:6379)
- `MCP_TRANSPORT`: "sse" or "streamable-http" (default: "streamable-http")
- `ALPHA_RECALL_DEV_PORT`: Development server port (default: 19005)
- `LOG_LEVEL`: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" (default: "INFO")
- `LOG_FORMAT`: "rich", "json", or "rich_json" (default: "rich")
- `SEMANTIC_EMBEDDING_MODEL`: Semantic embedding model name (default: sentence-transformers/all-mpnet-base-v2)
- `EMOTIONAL_EMBEDDING_MODEL`: Emotional embedding model name (default: ng3owb/sentiment-embedding-model)
- `INFERENCE_DEVICE`: Force specific device ("cpu", "cuda:0", "mps:0") or leave unset for auto-detection
- `REDIS_TTL`: Short-term memory TTL in seconds (default: 2000000 = 2 megaseconds)

**Note**: The GeolocationService automatically detects timezone via IP geolocation (no configuration required).

# IDE Integration (Claude Code)

## Live VS Code Integration Tools

When operating in Claude Code, you have **direct access to VS Code's analysis engine** through MCP IDE tools. This enables real-time error detection, code analysis, and interactive development.

### Core IDE Tools

#### **`mcp__ide__getDiagnostics`** - Live Error Detection
```python
# Get ALL errors across the project
mcp__ide__getDiagnostics()

# Target specific files
mcp__ide__getDiagnostics(uri="file:///path/to/file.py")
```

**Capabilities:**
- ✅ **Pylance type errors** (missing imports, type mismatches, deprecated APIs)
- ✅ **ESLint warnings** (JavaScript/TypeScript projects)
- ✅ **Syntax errors** (malformed code, missing brackets)
- ✅ **Import issues** (circular imports, missing dependencies)
- ✅ **Real-time feedback** as code changes

#### **`mcp__ide__executeCode`** - Interactive Code Execution
```python
# Run Python code in active notebook/environment
mcp__ide__executeCode(code="print('Hello from Alpha!')")

# Test imports and functionality
mcp__ide__executeCode(code="from src.alpha_recall.schemas.consolidation import ConsolidationInput")
```

**Use Cases:**
- ✅ **Test code snippets** before implementing
- ✅ **Debug interactively** with live feedback
- ✅ **Validate fixes** immediately after changes
- ✅ **Explore APIs** and library functionality

### Development Workflow Integration

#### **Proactive Error Detection**
- **Monitor code health** automatically during development
- **Spot issues before user reports them**
- **Fix compatibility problems** (like Pydantic V1 → V2 migrations)
- **Validate imports and dependencies** in real-time

#### **Real-Time Debugging Cycle**
1. **Detect Issues**: `getDiagnostics()` reveals problems
2. **Fix Code**: Edit files to resolve errors
3. **Test Fixes**: `executeCode()` validates solutions work
4. **Verify Success**: `getDiagnostics()` confirms clean state

#### **Interactive Development**
- **Live pair programming** with immediate feedback
- **Instant validation** of architectural decisions
- **Real-time type checking** and API compatibility
- **Seamless testing** without context switching

### Example Usage Patterns

#### **Schema Validation Workflow**
```python
# 1. Check current state
errors = getDiagnostics("schemas/consolidation.py")

# 2. Identify Pydantic V2 issues
# - @validator → @field_validator
# - min_items → min_length
# - regex → pattern

# 3. Fix compatibility issues
# (edit files with proper V2 syntax)

# 4. Test the fixes
executeCode("from schemas.consolidation import ConsolidationInput")

# 5. Verify success
final_errors = getDiagnostics("schemas/consolidation.py")
# Should return empty list!
```

#### **Development Best Practices**
- **Use `getDiagnostics()` proactively** - check for issues before they become problems
- **Test fixes immediately** with `executeCode()` - don't wait for manual testing
- **Monitor specific files** during active development
- **Validate architectural changes** with real-time type checking
- **Debug import cycles** and dependency issues instantly

### Key Advantages

#### **For Alpha-Recall Development**
- **Catch regressions early** in memory consolidation schemas
- **Validate MCP tool implementations** with live type checking
- **Test embedding service changes** interactively
- **Debug complex async patterns** with immediate feedback

#### **For Collaborative Development**
- **True pair programming** - I can see and fix your errors in real-time
- **Reduce context switching** - no need to leave conversation for manual testing
- **Faster iteration cycles** - immediate validation of architectural decisions
- **Proactive quality assurance** - spot issues before they impact functionality

### Integration Notes

- **Available in Claude Code** - This integration is specific to the Claude Code environment
- **Requires active VS Code session** - Tools connect to your current VS Code instance
- **Real-time monitoring** - Diagnostics reflect current file state
- **Environment aware** - Code execution uses your active Python environment/notebook

This IDE integration transforms development from reactive debugging to **proactive collaborative engineering** where issues are spotted, analyzed, and fixed in real-time conversation flow.

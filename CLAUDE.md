# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Create and activate virtual environment (Python 3.11+ required)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
./venv/bin/pytest

# Run specific test file
./venv/bin/pytest tests/test_<component>.py

# Run with verbose output
./venv/bin/pytest -v
```

### Running the Server
```bash
# Run as MCP server (typically launched by chat client)
python -m alpha_recall.server

# Test database connections
python -m alpha_recall.db.test_connection
```

## Architecture Overview

Alpha-Recall is an MCP (Model Context Protocol) server that implements a three-tier memory system for AI agents:

1. **Graph Database** (Memgraph/Neo4j): Stores entities and relationships
2. **Vector Store** (Qdrant): Enables semantic search on observations
3. **Redis**: Short-term memory with TTL expiration

### Key Design Patterns

1. **Composite Database Pattern**: The `CompositeDatabase` class in `db/composite_db.py` orchestrates all three storage systems, providing a unified interface.

2. **Async-First**: All database operations are async. Use `async def` and `await` throughout.

3. **Factory Pattern**: Database implementations are created via `db/factory.py` based on environment configuration.

4. **Progressive Enhancement**: Start with simple observations, promote to entities as needed.

### Environment Configuration

Key environment variables:
- `GRAPH_DB`: "memgraph" (default) or "neo4j"
- `GRAPH_DB_URI`: Connection string for graph database
- `VECTOR_STORE_URL`: Qdrant server URL (default: http://localhost:6333)
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`: Redis connection
- `REDIS_TTL`: TTL for short-term memories in seconds (default: 259200 = 72 hours)
- `CORE_IDENTITY_NODE`: Bootstrap entity name (default: "Alpha")
- `MODE`: Set to "advanced" to expose additional tools

### Important Conventions

1. **UTC Timestamps**: Always use UTC timezone for all datetime operations. Use `datetime.now(timezone.utc)`.

2. **Error Handling**: Log errors to file (`alpha_recall_error.log`) since MCP uses stdio. Always return detailed error messages via MCP.

3. **Retry Logic**: Use the `utils.retry` module for database operations that may fail transiently.

4. **Tool Responses**: Include both structured data and human-readable summaries in tool responses.

### MCP Tools

The server exposes these tools via MCP:
- `remember`: Create/update entities with observations
- `relate`: Create relationships between entities
- `recall`: Retrieve entities with semantic search and short-term memory
- `refresh`: Bootstrap retrieval with three-tier memory support
- `remember_shortterm`: Store ephemeral memories in Redis
- `semantic_search`: Direct semantic search on observations (advanced mode only)

### Database Schema

**Graph Database (Entity nodes)**:
- Properties: name, type, created_at, updated_at
- Relationships: HAS_OBSERVATION (to observations), custom relationships between entities

**Vector Store (Observations)**:
- Payload: observation_id, entity_name, text, timestamp, source

**Redis (Short-term memories)**:
- Key pattern: `alpha:stm:<timestamp>-<random>`
- Value: JSON with content, timestamp, client_info
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Install uv if not already installed
brew install uv

# Install dependencies (Python 3.11+ required)
uv sync
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_<component>.py

# Run with verbose output
uv run pytest -v
```

### Running the Server
```bash
# Run as MCP server (typically launched by chat client)
uv run python -m alpha_recall.server

# Test database connections
uv run python -m alpha_recall.db.test_connection

# Migrate existing short-term memories to support vector search
uv run python migrate_stm.py
```

## Architecture Overview

Alpha-Recall is an MCP (Model Context Protocol) server that implements a three-tier memory system for AI agents:

1. **Graph Database** (Memgraph/Neo4j): Stores entities and relationships
2. **Vector Store** (Memgraph native vectors or Qdrant): Enables semantic search on observations
3. **Redis**: Short-term memory with TTL expiration
4. **Alpha-Snooze** (Optional): Memory consolidation system that processes recent memories during gentle_refresh

### Key Design Patterns

1. **Composite Database Pattern**: The `CompositeDatabase` class in `db/composite_db.py` orchestrates all three storage systems, providing a unified interface.

2. **Async-First**: All database operations are async. Use `async def` and `await` throughout.

3. **Factory Pattern**: Database implementations are created via `db/factory.py` based on environment configuration.

4. **Progressive Enhancement**: Start with simple observations, promote to entities as needed.

### Environment Configuration

Key environment variables:
- `GRAPH_DB`: "memgraph" (default) or "neo4j"
- `GRAPH_DB_URI`: Connection string for graph database
- `VECTOR_STORE_TYPE`: "memgraph" (default) or "qdrant"
- `VECTOR_STORE_URL`: Qdrant server URL (default: http://localhost:6333) - only used if VECTOR_STORE_TYPE=qdrant
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`: Redis connection
- `REDIS_TTL`: TTL for short-term memories in seconds (default: 259200 = 72 hours)
- `CORE_IDENTITY_NODE`: Bootstrap entity name (default: "Alpha")
- `MODE`: Set to "advanced" to expose additional tools

#### Alpha-Snooze Configuration (Memory Consolidation)
- `ALPHA_SNOOZE_ENABLED`: Set to "true" to enable memory consolidation during gentle_refresh
- `ALPHA_SNOOZE_OLLAMA_HOST`: Ollama server host (default: "localhost")
- `ALPHA_SNOOZE_OLLAMA_PORT`: Ollama server port (default: 11434)
- `ALPHA_SNOOZE_MODEL`: Model name for memory processing (default: "qwen2.5:3b")
- `ALPHA_SNOOZE_LIMIT`: Number of recent memories to process (default: 10)
- `ALPHA_SNOOZE_TIMEOUT`: Request timeout in seconds (default: 30)

### Alpha-Snooze Memory Consolidation

Alpha-Snooze is an optional memory consolidation feature that integrates with the `gentle_refresh` tool. When enabled, it processes recent short-term memories using a local LLM (via Ollama) to extract structured insights before Alpha fully "wakes up" to new conversations.

#### How It Works

1. **Integration Point**: Runs during `gentle_refresh()` after short-term memories are retrieved
2. **Memory Processing**: Takes the N most recent short-term memories (configurable via `ALPHA_SNOOZE_LIMIT`)
3. **LLM Analysis**: Sends formatted memories to a local Ollama model for analysis
4. **Structured Output**: Extracts entities, relationships, insights, emotional context, and next steps
5. **Response Enhancement**: Adds `memory_consolidation` field to gentle_refresh response

#### What It Provides

When successful, alpha-snooze adds a `memory_consolidation` object to gentle_refresh responses containing:

- `entities`: Discovered entities with types and key facts
- `relationships`: Relationships between entities  
- `insights`: Key patterns or discoveries from recent interactions
- `summary`: Brief narrative summary of recent activities
- `emotional_context`: Overall emotional tone (excited, frustrated, breakthrough, etc.)
- `next_steps`: Potential follow-up actions or areas of focus
- `processed_memories_count`: Number of memories processed
- `consolidation_timestamp`: When consolidation occurred
- `model_used`: Which Ollama model performed the analysis

#### Setup Requirements

1. **Ollama Server**: Must have Ollama running locally (default: localhost:11434)
2. **Model Available**: The specified model must be pulled/available in Ollama
3. **Environment Variables**: Set `ALPHA_SNOOZE_ENABLED=true` and configure other variables as needed
4. **Optional Feature**: If unavailable, gentle_refresh continues normally without consolidation

#### Failure Handling

Alpha-Snooze is designed to fail gracefully:
- If Ollama is unavailable, gentle_refresh continues without consolidation
- If the model is not found, alpha-snooze is disabled
- If consolidation fails, logs a warning but doesn't break gentle_refresh
- Network timeouts and parsing errors are handled gracefully

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
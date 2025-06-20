# Alpha-Recall

A sophisticated three-tier memory system for AI agents, implementing persistent memory through graph databases, semantic search, and short-term memory with TTL expiration. Built for the Model Context Protocol (MCP).

## Overview

Alpha-Recall provides AI agents with a human-like memory architecture that enables:
- **Persistent long-term memory** via knowledge graphs
- **Semantic search** across memories using vector embeddings
- **Emotional search** to find memories by emotional resonance
- **Short-term memory** with automatic expiration
- **Cross-session continuity** for maintaining context

## Architecture

### Three-Tier Memory System

1. **Graph Database Layer** (Memgraph/Neo4j)
   - Stores entities and their relationships
   - Maintains structured knowledge as a graph
   - Supports both Memgraph and Neo4j backends

2. **Vector Store Layer** (Qdrant)
   - Enables semantic search using sentence embeddings
   - Default model: `all-MiniLM-L6-v2` (384 dimensions)
   - Stores observation embeddings for similarity search

3. **Short-Term Memory Layer** (Redis)
   - Ephemeral memories with configurable TTL (default: 72 hours)
   - Supports both semantic and emotional vector search
   - Emotional embeddings: 1024-dimensional vectors
   - Maintains recent context across sessions

### Key Components

- **CompositeDatabase**: Orchestrates all three storage systems
- **MCP Server**: Exposes memory operations as tools
- **Embedding Service**: HTTP microservice for generating embeddings
- **Migration Scripts**: Upgrade existing data to new formats

## Installation

```bash
# Install uv package manager
brew install uv

# Clone the repository
git clone <repository-url>
cd alpha-recall

# Install dependencies
uv sync
```

## Configuration

Create a `.env` file with the following variables:

```bash
# Graph Database
GRAPH_DB=memgraph  # or "neo4j"
GRAPH_DB_URI=bolt://localhost:7687

# Vector Store
VECTOR_STORE_URL=http://localhost:6333

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # optional
REDIS_TTL=259200  # 72 hours in seconds

# Embedding Services
EMBEDDING_SERVER_URL=http://localhost:6004/encode
EMOTIONAL_EMBEDDING_URL=http://localhost:6004/sentiment-embeddings

# Core Configuration
CORE_IDENTITY_NODE=Alpha  # Bootstrap entity name
MODE=  # Set to "advanced" for additional tools
```

## Usage

### Running the MCP Server

```bash
# Run as MCP server (typically launched by chat client)
uv run python -m alpha_recall.server
```

### Testing Connections

```bash
# Test database connections
uv run python -m alpha_recall.db.test_connection
```

### Running Migrations

```bash
# Migrate short-term memories to support vector search
uv run python migrate_stm.py

# Upgrade emotional embeddings from 7D to 1024D
uv run python migrate_emotional_vectors.py
```

## MCP Tools

The server exposes the following tools via MCP:

### `refresh`
Bootstrap tool that loads three tiers of memory:
- Tier 1: Core identity information
- Tier 2: Recent short-term memories
- Tier 3: Contextually relevant memories based on query

```python
# Usage
await refresh(query="Hello, how are you today?")
```

### `remember`
Create or update entities with observations:
```python
await remember(
    entity="Project Alpha",
    entity_type="Project",
    observation="A research project exploring persistent AI identity"
)
```

### `relate`
Create relationships between entities:
```python
await relate(
    entity="Alpha",
    to_entity="Jeffery Harrell",
    as_type="was created by"
)
```

### `recall`
Retrieve memories with two modes:

**Long-term memory mode** (default):
```python
await recall(
    query="Project Alpha",  # Entity name or semantic search
    depth=1  # Relationship hops to include
)
```

**Short-term memory mode**:
```python
await recall(
    shortterm=True,
    through_the_last="2h"  # Optional time filter
)
```

### `remember_shortterm`
Store ephemeral memories that expire after TTL:
```python
await remember_shortterm(
    content="Just completed code review with the team"
)
# Returns the stored memory plus 5 most relevant memories
```

### Advanced Tools (MODE=advanced)

- `semantic_search`: Direct semantic search on observations
- `recency_search`: Get N most recent observations

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_emotional_embeddings.py

# Run with coverage
uv run pytest --cov
```

### Project Structure

```
alpha_recall/
├── src/alpha_recall/
│   ├── db/                 # Database implementations
│   │   ├── base.py        # Abstract base classes
│   │   ├── composite_db.py # Orchestrates all DBs
│   │   ├── memgraph_db.py # Memgraph implementation
│   │   ├── neo4j_db.py    # Neo4j implementation
│   │   ├── redis_db.py     # Redis short-term memory
│   │   └── vector_store.py # Qdrant implementation
│   ├── models/            # Data models
│   ├── tools/             # MCP tool implementations
│   ├── utils/             # Utilities (retry logic, etc.)
│   └── server.py          # Main MCP server
├── tests/                 # Test suite
├── docs/                  # Documentation and ADRs
└── migrate_*.py          # Migration scripts
```

## Key Features

### Semantic + Emotional Search
The system supports dual vector search:
- **Semantic search**: Finds memories by meaning similarity
- **Emotional search**: Finds memories by emotional resonance

### Automatic Retry Logic
Database operations include configurable retry logic for handling transient failures.

### UTC Timestamps
All timestamps use UTC timezone for consistency across systems.

### Error Logging
Errors are logged to `alpha_recall_error.log` since MCP uses stdio for communication.

## Architecture Decision Records

See the `docs/` directory for ADRs documenting key design decisions:
- ADR-007: Enhanced Recall with Integrated Semantic Search
- ADR-008: Enhanced Bootstrap with Semantic Greeting Analysis
- ADR-009: Redis-Based Short-Term Memory System

## License

[License information here]

## Contributing

[Contributing guidelines here]
# Alpha-Recall Architecture

## System Overview

Alpha-Recall implements a sophisticated memory system inspired by human cognitive architecture. The system combines three distinct storage mechanisms to provide both immediate context retention and long-term knowledge persistence.

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Client                            │
│                   (Claude Desktop, etc.)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol
┌─────────────────────────┴───────────────────────────────────┐
│                     MCP Server (server.py)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Tools: refresh, recall, remember, relate, etc.      │   │
│  └─────────────────────────┬───────────────────────────┘   │
└────────────────────────────┼───────────────────────────────┘
                             │
┌────────────────────────────┴───────────────────────────────┐
│                    CompositeDatabase                        │
│  Orchestrates three storage systems with unified interface  │
└──────┬─────────────────────┬─────────────────────┬─────────┘
       │                     │                     │
┌──────┴────────┐     ┌──────┴────────┐    ┌──────┴────────┐
│ Graph Database│     │ Vector Store  │    │ Redis STM     │
│ (Memgraph/    │     │ (Qdrant)      │    │ (Short-Term)  │
│  Neo4j)       │     │               │    │               │
├───────────────┤     ├───────────────┤    ├───────────────┤
│ • Entities    │     │ • Semantic    │    │ • Recent      │
│ • Relations   │     │   embeddings  │    │   memories    │
│ • Observations│     │ • 384D vectors│    │ • 72hr TTL    │
│ • Knowledge   │     │ • Similarity  │    │ • Semantic &  │
│   graph       │     │   search      │    │   emotional   │
└───────────────┘     └───────────────┘    └───────────────┘
```

## Component Details

### 1. MCP Server Layer

The server exposes memory operations as Model Context Protocol tools:

- **Entry Point**: `server.py`
- **Lifecycle Management**: Handles startup/shutdown via `server_lifespan`
- **Tool Registration**: Decorates async functions with `@mcp.tool`
- **Error Handling**: Comprehensive try/catch with detailed logging

### 2. CompositeDatabase

Central orchestrator that provides a unified interface across all storage systems:

```python
class CompositeDatabase:
    def __init__(self, graph_db, semantic_search, shortterm_memory):
        self.graph_db = graph_db          # GraphDatabase implementation
        self.search_engine = semantic_search  # SemanticSearch implementation
        self.shortterm_memory = shortterm_memory  # RedisShortTermMemory
```

Key responsibilities:
- Coordinates multi-database operations
- Ensures consistency across stores
- Handles fallback strategies when features unavailable

### 3. Graph Database Layer

Stores structured knowledge as entities and relationships.

**Supported Backends**:
- Memgraph (default)
- Neo4j

**Schema**:
```cypher
// Entity node
(:Entity {
    id: UUID,
    name: String,
    type: String,
    created_at: DateTime,
    updated_at: DateTime
})

// Observation node
(:Observation {
    id: UUID,
    content: String,
    created_at: DateTime
})

// Relationships
(entity:Entity)-[:HAS_OBSERVATION]->(obs:Observation)
(entity1:Entity)-[:CUSTOM_RELATION]->(entity2:Entity)
```

### 4. Vector Store Layer

Enables semantic search using sentence embeddings.

**Implementation**: Qdrant vector database
**Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
**Distance Metric**: Cosine similarity

**Storage Format**:
```json
{
    "observation_id": "uuid",
    "vector": [0.1, 0.2, ...],  // 384D
    "payload": {
        "entity_name": "Alpha",
        "text": "observation content",
        "created_at": "2025-06-20T..."
    }
}
```

### 5. Short-Term Memory Layer

Redis-based ephemeral storage with advanced search capabilities.

**Features**:
- Configurable TTL (default: 72 hours)
- Dual vector search:
  - Semantic embeddings (384D)
  - Emotional embeddings (1024D)
- Client tracking for multi-interface support

**Data Structure**:
```
Key: alpha:stm:{timestamp}-{uuid}
Type: Hash
Fields:
  - content: String
  - created_at: ISO timestamp
  - client: JSON (client metadata)
  - embedding: Binary (384D float32 array)
  - embedding_emotional: Binary (1024D float32 array)
```

**Vector Indices**:
```
idx:stm - Semantic search index
idx:stm_emotional - Emotional search index
```

## Data Flow Examples

### 1. Memory Bootstrap (refresh)

```
User greeting → refresh tool
    ├── Load core identity (Graph DB)
    ├── Retrieve recent STM (Redis)
    └── Semantic search on greeting (Qdrant + Redis)
        └── Return merged, deduplicated results
```

### 2. Creating a Memory (remember)

```
New observation → remember tool
    ├── Create/update entity (Graph DB)
    ├── Store observation (Graph DB)
    └── Index observation (Qdrant)
```

### 3. Short-term Memory Storage

```
Memory content → remember_shortterm tool
    ├── Generate embeddings (HTTP service)
    │   ├── Semantic (384D)
    │   └── Emotional (1024D)
    ├── Store in Redis with TTL
    └── Return relevant memories
        ├── Semantic search
        └── Emotional search
```

## Embedding Services

The system relies on external HTTP services for embedding generation:

### Semantic Embeddings
- **Endpoint**: `http://localhost:6004/encode`
- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Purpose**: Meaning-based similarity

### Emotional Embeddings
- **Endpoint**: `http://localhost:6004/sentiment-embeddings`
- **Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Dimensions**: 1024
- **Purpose**: Emotional resonance matching

## Retry and Error Handling

The system implements sophisticated retry logic:

```python
@async_retry(
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0,
    error_messages_to_retry=["failed to receive chunk size"]
)
```

This handles transient failures in:
- Database connections
- Network requests
- Embedding generation

## Performance Considerations

1. **Vector Search Optimization**
   - Redis vector indices for O(log n) search
   - Qdrant's HNSW algorithm for approximate nearest neighbors

2. **Connection Pooling**
   - Reused database connections via lifespan management
   - HTTP client connection reuse for embeddings

3. **Memory Limits**
   - Semantic search limited to top N results
   - Query truncation at 1000 characters for bootstrap

## Security Considerations

1. **Authentication**
   - Redis password support
   - Graph database authentication via URI

2. **Data Isolation**
   - Configurable key prefixes in Redis
   - Database selection support

3. **Network Security**
   - All services expected on private network
   - No built-in TLS (rely on network security)

## Future Enhancements

1. **Planned Features**
   - Multi-modal embeddings (image, audio)
   - Hierarchical memory organization
   - Memory importance scoring

2. **Scalability**
   - Distributed Redis cluster support
   - Graph database sharding
   - Vector index partitioning

3. **Observability**
   - Metrics collection
   - Distributed tracing
   - Performance monitoring
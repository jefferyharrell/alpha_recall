# Development Notes

- Prefer `pathlib` over `os.path`.

## Environment Variables

### Database Configuration
- `GRAPH_DB`: Type of graph database to use (default: "neo4j", future: "memgraph")
- `GRAPH_DB_URI`: Connection URI for the graph database
- `GRAPH_DB_USER`: Username for the graph database
- `GRAPH_DB_PASSWORD`: Password for the graph database

### Logging Configuration
- `LOG_FILE`: Path to the log file
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Architecture Notes

### Database Layer
- Pluggable architecture with abstract GraphDatabase base class
- Neo4j implementation for MVP
- Factory pattern for creating database instances based on environment variables
- Future support for other graph databases like Memgraph

### Logging
- Uses pathlib for all path operations
- Creates log directories at runtime if they don't exist
- Configurable log level via environment variables
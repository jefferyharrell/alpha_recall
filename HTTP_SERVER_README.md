# Alpha-Recall HTTP/SSE Server

This feature branch implements an HTTP server with Server-Sent Events (SSE) transport for the alpha-recall MCP server.

## Overview

The HTTP server provides:
- RESTful endpoints for tool invocation
- SSE endpoint for real-time streaming communication
- Docker containerization with hot reload
- CORS support for web clients

## Quick Start

### 1. Install dependencies (if running locally)
```bash
uv sync
```

### 2. Run with Docker Compose (recommended)
```bash
docker-compose up --build
```

The server will be available at `http://localhost:8080`

### 3. Test the server
```bash
python test_http_server.py
```

## API Endpoints

### Health Check
```
GET /health
```

### List Available Tools
```
GET /tools
```

### Call a Tool
```
POST /tool/{tool_name}
Content-Type: application/json

{
  "tool": "refresh",
  "params": {
    "query": "Hello!"
  }
}
```

### SSE Stream
```
GET /sse
```

### SSE Tool Call
```
POST /sse/call
Content-Type: application/json

{
  "tool": "refresh",
  "params": {
    "query": "Hello!"
  }
}
```

## Development

The Docker container mounts the source code, so changes to Python files will trigger automatic reload.

## Environment Variables

- `HTTP_HOST`: Host to bind to (default: 0.0.0.0)
- `HTTP_PORT`: Port to bind to (default: 8080)
- `MODE`: Set to "advanced" to enable advanced tools

## Next Steps

1. Create Node.js/TypeScript client SDK
2. Implement proper SSE message protocol
3. Add authentication/authorization
4. Integrate with the full database stack
5. Add request queuing and rate limiting

## Notes

- Currently configured to connect to database services on the host machine
- Tool calls will fail without proper database connections
- This is intentional for isolated testing of the HTTP transport layer
"""
FastAPI HTTP server with SSE transport for alpha-recall MCP.

This module provides an HTTP/SSE interface to the alpha-recall MCP server,
allowing clients to connect via HTTP instead of stdio transport.
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from alpha_recall.db import create_db_instance
from alpha_recall.logging_utils import configure_logging, get_logger
from alpha_recall.server import (
    ask_memory,
    list_narratives,
    recall,
    recall_narrative,
    recency_search,
    refresh,
    relate,
    remember,
    remember_narrative,
    remember_shortterm,
    search_narratives,
    semantic_search,
)

# Load environment variables
load_dotenv()

# Configure logging
logger = configure_logging()
logger = get_logger("http_server")

# Server configuration
SERVER_NAME = "alpha-recall-http"
HTTP_PORT = int(os.environ.get("HTTP_PORT", 8080))
HTTP_HOST = os.environ.get("HTTP_HOST", "0.0.0.0")

# Global database connection
db_instance = None


class MCPRequest(BaseModel):
    """Base request model for MCP tool calls."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)


class MCPResponse(BaseModel):
    """Response model for MCP tool calls."""

    id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SSEMessage(BaseModel):
    """Server-sent event message format."""

    type: str  # "tool_result", "error", "ping"
    data: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - database connections, etc.
    """
    global db_instance
    logger.info(f"Starting {SERVER_NAME} HTTP server")

    try:
        # Initialize database connection
        db_instance = await create_db_instance()
        logger.info("Database connection established")
        yield
    except Exception as e:
        logger.error(f"Error during server startup: {str(e)}")
        raise
    finally:
        # Clean up resources on shutdown
        if db_instance:
            logger.info("Closing database connection")
            await db_instance.close()
        logger.info(f"Shutting down {SERVER_NAME} HTTP server")


# Create FastAPI app
app = FastAPI(
    title="Alpha-Recall HTTP Server",
    description="HTTP/SSE interface for alpha-recall MCP server",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create a mock context object that mimics MCP context
class MockContext:
    """Mock context to simulate MCP context for tool functions."""

    def __init__(self, db):
        self.db = db
        self.lifespan_context = type("obj", (object,), {"db": db})


# Tool registry mapping tool names to functions
TOOL_REGISTRY = {
    "recall": recall,
    "refresh": refresh,
    "remember": remember,
    "remember_shortterm": remember_shortterm,
    "relate": relate,
    "semantic_search": semantic_search,
    "remember_narrative": remember_narrative,
    "search_narratives": search_narratives,
    "recall_narrative": recall_narrative,
    "list_narratives": list_narratives,
}

# Add advanced tools if mode is set
if os.environ.get("MODE", "").lower() == "advanced":
    TOOL_REGISTRY["recency_search"] = recency_search

# Log registered tools for debugging
logger.info(f"Registered MCP tools: {list(TOOL_REGISTRY.keys())}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "server": SERVER_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/tools")
async def list_tools():
    """List available MCP tools."""
    tools = []
    for name, func in TOOL_REGISTRY.items():
        tools.append(
            {
                "name": name,
                "description": func.__doc__.strip() if func.__doc__ else None,
            }
        )
    return {"tools": tools}


@app.post("/tool/{tool_name}")
async def call_tool(tool_name: str, request: MCPRequest):
    """
    Call an MCP tool directly via HTTP POST.
    
    This endpoint is for simple request/response interactions.
    For streaming, use the SSE endpoint.
    """
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    if request.tool != tool_name:
        raise HTTPException(
            status_code=400, detail="Tool name in URL and request body must match"
        )

    try:
        # Create mock context with database
        ctx = MockContext(db_instance)

        # Get the tool function
        tool_func = TOOL_REGISTRY[tool_name]

        # Call the tool with the provided parameters
        result = await tool_func(ctx, **request.params)

        return MCPResponse(
            id=request.id, success=result.get("success", True), result=result
        )
    except TypeError as e:
        # Handle parameter mismatch errors
        error_msg = f"Invalid parameters for tool '{tool_name}': {str(e)}"
        logger.error(error_msg)
        return MCPResponse(id=request.id, success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        logger.error(error_msg)
        return MCPResponse(id=request.id, success=False, error=error_msg)


async def sse_generator(request: Request):
    """
    Generate server-sent events for MCP communication.
    
    This supports the full MCP protocol over SSE transport.
    """
    logger.info("New SSE connection established")
    
    try:
        # Send initial ping to confirm connection
        yield {
            "event": "message",
            "data": json.dumps(
                SSEMessage(
                    type="ping",
                    data={"timestamp": datetime.now(timezone.utc).isoformat()},
                ).model_dump()
            ),
        }

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("SSE client disconnected")
                break

            # Wait for messages (this would be replaced with actual message queue)
            await asyncio.sleep(30)  # Send ping every 30 seconds
            
            # Send periodic ping to keep connection alive
            yield {
                "event": "message",
                "data": json.dumps(
                    SSEMessage(
                        type="ping",
                        data={"timestamp": datetime.now(timezone.utc).isoformat()},
                    ).model_dump()
                ),
            }

    except asyncio.CancelledError:
        logger.info("SSE connection cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in SSE generator: {str(e)}")
        yield {
            "event": "error",
            "data": json.dumps(
                SSEMessage(type="error", data={"error": str(e)}).model_dump()
            ),
        }


@app.get("/sse")
async def sse_endpoint(request: Request):
    """
    Server-sent events endpoint for MCP protocol over HTTP.
    
    This endpoint maintains a persistent connection and streams
    MCP messages using the SSE protocol.
    """
    return EventSourceResponse(sse_generator(request))


@app.post("/sse")
async def sse_post_endpoint(request: Request):
    """
    Handle POST requests to SSE endpoint for MCP message sending.
    
    This is used by mcp-remote and other clients that need to send
    MCP messages via POST while maintaining SSE connection.
    """
    try:
        # Read the request body
        body = await request.body()
        mcp_message = json.loads(body.decode())
        
        # Process the MCP message
        if "method" in mcp_message:
            method = mcp_message["method"]
            params = mcp_message.get("params", {})
            message_id = mcp_message.get("id")
            
            # Handle MCP methods
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "resources": {},
                            "prompts": {},
                        },
                        "serverInfo": {
                            "name": SERVER_NAME,
                            "version": "0.1.0",
                        },
                    },
                }
            elif method == "tools/list":
                tools = []
                for name, func in TOOL_REGISTRY.items():
                    tools.append({
                        "name": name,
                        "description": func.__doc__.strip() if func.__doc__ else None,
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    })
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {"tools": tools},
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                if tool_name in TOOL_REGISTRY:
                    try:
                        ctx = MockContext(db_instance)
                        tool_func = TOOL_REGISTRY[tool_name]
                        result = await tool_func(ctx, **tool_args)
                        
                        return {
                            "jsonrpc": "2.0",
                            "id": message_id,
                            "result": {
                                "content": [{"type": "text", "text": json.dumps(result)}],
                            },
                        }
                    except Exception as e:
                        return {
                            "jsonrpc": "2.0",
                            "id": message_id,
                            "error": {
                                "code": -32603,
                                "message": f"Tool execution error: {str(e)}",
                            },
                        }
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "error": {
                            "code": -32601,
                            "message": f"Tool '{tool_name}' not found",
                        },
                    }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not implemented",
                    },
                }
        else:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid Request - missing method",
                },
            }
            
    except json.JSONDecodeError:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32700,
                "message": "Parse error - invalid JSON",
            },
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}",
            },
        }


@app.post("/sse/call")
async def sse_call_tool(request: MCPRequest):
    """
    Call a tool and return the result for SSE clients.
    
    This is used by SSE clients to invoke tools while maintaining
    their SSE connection for receiving events.
    """
    tool_name = request.tool
    
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        # Create mock context with database
        ctx = MockContext(db_instance)

        # Get the tool function
        tool_func = TOOL_REGISTRY[tool_name]

        # Call the tool with the provided parameters
        result = await tool_func(ctx, **request.params)

        return MCPResponse(
            id=request.id, success=result.get("success", True), result=result
        )
    except TypeError as e:
        # Handle parameter mismatch errors
        error_msg = f"Invalid parameters for tool '{tool_name}': {str(e)}"
        logger.error(error_msg)
        return MCPResponse(id=request.id, success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        logger.error(error_msg)
        return MCPResponse(id=request.id, success=False, error=error_msg)


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "alpha_recall.http_server:app",
        host=HTTP_HOST,
        port=HTTP_PORT,
        reload=True,  # Enable hot reload for development
        log_level="info",
    )
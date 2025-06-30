"""Unit tests for health check functionality."""

import json
from datetime import datetime, timezone

import pytest

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.health import register_health_tools
from alpha_recall.version import __version__


class MockMCP:
    """Mock MCP server for testing tool registration."""
    
    def __init__(self):
        self.tools = {}
    
    def tool(self):
        """Decorator to register a tool."""
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator


def test_health_check_registration():
    """Test that health check tool registers properly."""
    mock_mcp = MockMCP()
    register_health_tools(mock_mcp)
    
    assert "health_check" in mock_mcp.tools
    assert callable(mock_mcp.tools["health_check"])


def test_health_check_response_structure():
    """Test health check returns proper JSON structure."""
    mock_mcp = MockMCP()
    register_health_tools(mock_mcp)
    
    health_check = mock_mcp.tools["health_check"]
    response = health_check()
    
    # Parse JSON response
    data = json.loads(response)
    
    # Verify required fields
    assert "status" in data
    assert "version" in data
    assert "checks" in data
    assert "timestamp" in data
    
    # Verify data types and values
    assert isinstance(data["status"], str)
    assert data["version"] == __version__
    assert isinstance(data["checks"], dict)
    assert isinstance(data["timestamp"], str)


def test_health_check_status_ok():
    """Test health check returns OK status when all checks pass."""
    mock_mcp = MockMCP()
    register_health_tools(mock_mcp)
    
    health_check = mock_mcp.tools["health_check"]
    response = health_check()
    data = json.loads(response)
    
    # For now, all checks should pass (they're just placeholders)
    assert data["status"] == "ok"
    assert "memgraph" in data["checks"]
    assert "redis" in data["checks"]
    assert data["checks"]["memgraph"] == "ok"
    assert data["checks"]["redis"] == "ok"


def test_health_check_timestamp_format():
    """Test health check timestamp is valid ISO format."""
    mock_mcp = MockMCP()
    register_health_tools(mock_mcp)
    
    health_check = mock_mcp.tools["health_check"]
    response = health_check()
    data = json.loads(response)
    
    # Should be able to parse as ISO datetime
    timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
    
    # Should be recent (within last minute) and in UTC
    now = datetime.now(timezone.utc)
    time_diff = abs((now - timestamp).total_seconds())
    assert time_diff < 60  # Within 1 minute
    assert timestamp.tzinfo is not None  # Has timezone info


def test_health_check_version_matches():
    """Test health check returns current version."""
    mock_mcp = MockMCP()
    register_health_tools(mock_mcp)
    
    health_check = mock_mcp.tools["health_check"]
    response = health_check()
    data = json.loads(response)
    
    assert data["version"] == __version__


def test_health_check_json_validity():
    """Test health check returns valid JSON."""
    mock_mcp = MockMCP()
    register_health_tools(mock_mcp)
    
    health_check = mock_mcp.tools["health_check"]
    response = health_check()
    
    # Should not raise any exception
    data = json.loads(response)
    
    # Should be able to serialize back to JSON
    json.dumps(data)  # This will raise if data contains non-serializable objects


@pytest.mark.parametrize("indent", [None, 2, 4])
def test_health_check_json_formatting(indent):
    """Test health check JSON is properly formatted."""
    mock_mcp = MockMCP()
    register_health_tools(mock_mcp)
    
    health_check = mock_mcp.tools["health_check"]
    response = health_check()
    
    # Should be properly formatted JSON (our implementation uses indent=2)
    data = json.loads(response)
    
    # Re-format with different indentation to verify structure
    reformatted = json.dumps(data, indent=indent)
    reparsed = json.loads(reformatted)
    
    # Should have same structure regardless of formatting
    assert reparsed == data
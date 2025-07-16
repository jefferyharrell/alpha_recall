"""Unit tests for gentle_refresh functionality.

Focused on behavior testing rather than format specifics.
Tests that the tool succeeds, returns plausible prose, and handles errors gracefully.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.gentle_refresh import (
    gentle_refresh,
    register_gentle_refresh_tools,
)


class MockMCP:
    """Mock MCP server for testing tool registration."""

    def __init__(self):
        self.tools = {}

    def tool(self, func=None):
        """Register a tool function or return decorator."""
        if func is not None:
            self.tools[func.__name__] = func
            return func
        else:

            def decorator(func):
                self.tools[func.__name__] = func
                return func

            return decorator


class MockTimeService:
    """Mock time service for testing."""

    def __init__(self):
        pass

    async def now_async(self):
        """Return mock time data."""
        return {
            "iso_datetime": "2025-01-01T12:00:00.000000+00:00",
            "local": "2025-01-01T07:00:00-05:00",
            "timezone": {
                "name": "America/Los_Angeles",
                "offset": "-08:00",
                "display": "PST",
            },
            "human_readable": "Wednesday, January 1, 2025 7:00 AM",
        }

    def format_datetime_for_model(self, dt):
        """Mock format datetime."""
        return "Wednesday, January 1, 2025 7:00 AM PST"


class MockGeolocationService:
    """Mock geolocation service for testing."""

    async def get_location(self):
        """Return mock location."""
        return "Los Angeles"


class MockSettings:
    """Mock settings for testing."""

    def __init__(self):
        self.gentle_refresh_default_tokens = 8000


class MockMemgraphDB:
    """Mock Memgraph database for testing."""

    def __init__(self, personality_data=None):
        self.personality_data = personality_data or []

    def execute_and_fetch(self, query):
        """Return mock personality data."""
        return self.personality_data


class MockMemgraphService:
    """Mock Memgraph service for testing."""

    def __init__(self, core_identity=None, personality_data=None):
        self.core_identity = core_identity or {"name": "Alpha Core Identity"}
        self.personality_data = personality_data or []
        self.db = MockMemgraphDB(personality_data)

    def get_core_identity(self):
        """Return mock core identity."""
        return self.core_identity


class MockRedisMemoryService:
    """Mock Redis memory service for testing."""

    def __init__(self):
        self.client = MockRedisClient()


class MockRedisIdentityService:
    """Mock Redis identity service for testing."""

    def __init__(self, identity_facts=None):
        self.identity_facts = identity_facts or []

    def get_identity_facts(self):
        """Return mock identity facts."""
        return self.identity_facts


class MockRedisContextService:
    """Mock Redis context service for testing."""

    def __init__(self, context_blocks=None):
        self.context_blocks = context_blocks or {}

    def get_all_context_blocks(self):
        """Return mock context blocks."""
        return {
            "success": True,
            "context_blocks": self.context_blocks,
        }

    def get_context_blocks_by_priority(self):
        """Return mock context blocks by priority (dict format with success key)."""
        # Convert context blocks to list format to match new structure
        context_blocks_list = []
        for key, content in self.context_blocks.items():
            context_blocks_list.append(
                {
                    "key": key,
                    "content": content,
                    "priority": 0.5,  # Default priority
                    "created_at": "2025-01-01T12:00:00.000000+00:00",
                    "updated_at": "2025-01-01T12:00:00.000000+00:00",
                }
            )
        return {
            "success": True,
            "context_blocks": context_blocks_list,
            "count": len(context_blocks_list),
            "operation_time_ms": 1.0,
        }

    def get_context_block(self, key):
        """Return mock context block for continuity message."""
        if key == "__continuity__":
            return {
                "success": False,
                "content": None,
                "has_content": False,
            }
        return {
            "success": True,
            "content": self.context_blocks.get(key, ""),
            "has_content": key in self.context_blocks,
        }


class MockRedisClient:
    """Mock Redis client for testing."""

    def zrevrange(self, key, start, end, withscores=False):
        """Mock zrevrange - return empty for simplicity."""
        return []

    def hmget(self, key, fields):
        """Mock hmget - return None for simplicity."""
        return [None, None, None]


def test_register_tools():
    """Test that tools are properly registered with MCP server."""
    mock_mcp = MockMCP()
    register_gentle_refresh_tools(mock_mcp)

    # Check that gentle_refresh is registered
    assert "gentle_refresh" in mock_mcp.tools
    assert callable(mock_mcp.tools["gentle_refresh"])


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.GeolocationService", MockGeolocationService)
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_context_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_identity_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_memory_service")
def test_gentle_refresh_basic_success(
    mock_redis_memory_service,
    mock_redis_identity_service,
    mock_redis_context_service,
    mock_memgraph_service,
):
    """Test that gentle_refresh returns valid prose on success."""
    # Setup mocks with BOTH identity facts AND personality data
    identity_facts = [
        {"content": "Alpha is an AI assistant", "score": 1.0, "position": 1}
    ]
    personality_data = [
        {
            "trait_name": "helpfulness",
            "trait_description": "Being helpful to users",
            "trait_weight": 1.0,
            "directive_instruction": "Always try to help",
            "directive_weight": 1.0,
        }
    ]

    mock_redis_memory_service.return_value = MockRedisMemoryService()
    mock_redis_identity_service.return_value = MockRedisIdentityService(
        identity_facts=identity_facts
    )
    mock_redis_context_service.return_value = MockRedisContextService()
    mock_memgraph_service.return_value = MockMemgraphService(
        personality_data=personality_data
    )

    response = asyncio.run(gentle_refresh())

    # Basic success criteria
    assert isinstance(response, str)
    assert len(response) > 100  # Should be substantial prose
    assert response.startswith("Good")
    assert "Los Angeles" in response
    assert "# Core Identity" in response
    assert "# Personality Traits" in response
    assert "# Recent Context" in response


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.GeolocationService", MockGeolocationService)
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_context_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_identity_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_memory_service")
def test_gentle_refresh_includes_identity_facts(
    mock_redis_memory_service,
    mock_redis_identity_service,
    mock_redis_context_service,
    mock_memgraph_service,
):
    """Test that identity facts appear in prose output."""
    # Setup with BOTH identity facts AND personality data
    identity_facts = [
        {"content": "Alpha is an AI", "score": 1.0, "position": 1},
        {"content": "Alpha helps with development", "score": 2.0, "position": 2},
    ]
    personality_data = [
        {
            "trait_name": "helpfulness",
            "trait_description": "Being helpful to users",
            "trait_weight": 1.0,
            "directive_instruction": "Always try to help",
            "directive_weight": 1.0,
        }
    ]

    mock_redis_memory_service.return_value = MockRedisMemoryService()
    mock_redis_identity_service.return_value = MockRedisIdentityService(
        identity_facts=identity_facts
    )
    mock_redis_context_service.return_value = MockRedisContextService()
    mock_memgraph_service.return_value = MockMemgraphService(
        personality_data=personality_data
    )

    response = asyncio.run(gentle_refresh())

    # Should include identity facts in prose
    assert "Alpha is an AI" in response
    assert "Alpha helps with development" in response


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.GeolocationService", MockGeolocationService)
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_context_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_identity_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_memory_service")
def test_gentle_refresh_includes_personality_traits(
    mock_redis_memory_service,
    mock_redis_identity_service,
    mock_redis_context_service,
    mock_memgraph_service,
):
    """Test that personality traits appear in prose output."""
    # Setup with BOTH identity facts AND personality data
    identity_facts = [
        {"content": "Alpha is an AI assistant", "score": 1.0, "position": 1}
    ]
    personality_data = [
        {
            "trait_name": "curiosity",
            "trait_description": "Being curious about things",
            "trait_weight": 0.9,
            "directive_instruction": "Ask lots of questions",
            "directive_weight": 0.8,
        }
    ]

    mock_redis_memory_service.return_value = MockRedisMemoryService()
    mock_redis_identity_service.return_value = MockRedisIdentityService(
        identity_facts=identity_facts
    )
    mock_redis_context_service.return_value = MockRedisContextService()
    mock_memgraph_service.return_value = MockMemgraphService(
        personality_data=personality_data
    )

    response = asyncio.run(gentle_refresh())

    # Should include personality info in prose
    assert "curiosity" in response.lower()
    assert "Being curious about things" in response


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.GeolocationService", MockGeolocationService)
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_context_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_identity_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_memory_service")
def test_gentle_refresh_handles_empty_data(
    mock_redis_memory_service,
    mock_redis_identity_service,
    mock_redis_context_service,
    mock_memgraph_service,
):
    """Test graceful handling when all data sources are empty."""
    # Setup with empty data - this should now return initialization error
    mock_redis_memory_service.return_value = MockRedisMemoryService()
    mock_redis_identity_service.return_value = MockRedisIdentityService(
        identity_facts=[]
    )
    mock_redis_context_service.return_value = MockRedisContextService()
    mock_memgraph_service.return_value = MockMemgraphService(
        core_identity={"name": "Alpha Core Identity", "observations": []},
        personality_data=[],
    )

    response = asyncio.run(gentle_refresh())

    # Should return initialization error when both identity facts and personality are missing
    assert isinstance(response, str)
    assert response.startswith("INITIALIZATION ERROR")
    assert "missing critical components" in response
    assert "identity facts (Redis)" in response
    assert "personality configuration (Memgraph)" in response


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.GeolocationService", MockGeolocationService)
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_context_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_identity_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_memory_service")
def test_gentle_refresh_error_resilience(
    mock_redis_memory_service,
    mock_redis_identity_service,
    mock_redis_context_service,
    mock_memgraph_service,
):
    """Test that gentle_refresh is resilient to partial service failures."""
    # Setup mocks where personality query fails but we have identity facts
    identity_facts = [
        {"content": "Alpha is an AI assistant", "score": 1.0, "position": 1}
    ]
    mock_memgraph = MockMemgraphService()
    mock_memgraph.db.execute_and_fetch = MagicMock(
        side_effect=Exception("Personality query failed")
    )
    mock_memgraph_service.return_value = mock_memgraph

    mock_redis_memory_service.return_value = MockRedisMemoryService()
    mock_redis_identity_service.return_value = MockRedisIdentityService(
        identity_facts=identity_facts
    )
    mock_redis_context_service.return_value = MockRedisContextService()

    response = asyncio.run(gentle_refresh())

    # Should return initialization error when personality fails (no personality data available)
    assert isinstance(response, str)
    assert response.startswith("INITIALIZATION ERROR")
    assert "missing critical components" in response
    assert "personality configuration (Memgraph)" in response


def test_gentle_refresh_with_token_budget():
    """Test that gentle_refresh accepts token budget parameter."""
    # This is a simple test that the function accepts the parameter
    # without testing the actual token limiting behavior
    with patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService()):
        with patch(
            "alpha_recall.tools.gentle_refresh.GeolocationService",
            MockGeolocationService,
        ):
            with patch("alpha_recall.tools.gentle_refresh.settings", MockSettings()):
                with patch(
                    "alpha_recall.tools.gentle_refresh.get_memgraph_service"
                ) as mock_memgraph:
                    with patch(
                        "alpha_recall.tools.gentle_refresh.get_redis_memory_service"
                    ) as mock_redis_memory:
                        with patch(
                            "alpha_recall.tools.gentle_refresh.get_redis_identity_service"
                        ) as mock_redis_identity:
                            with patch(
                                "alpha_recall.tools.gentle_refresh.get_redis_context_service"
                            ) as mock_redis_context:
                                # Setup with BOTH identity facts AND personality data
                                identity_facts = [
                                    {
                                        "content": "Alpha is an AI assistant",
                                        "score": 1.0,
                                        "position": 1,
                                    }
                                ]
                                personality_data = [
                                    {
                                        "trait_name": "helpfulness",
                                        "trait_description": "Being helpful to users",
                                        "trait_weight": 1.0,
                                        "directive_instruction": "Always try to help",
                                        "directive_weight": 1.0,
                                    }
                                ]
                                mock_memgraph.return_value = MockMemgraphService(
                                    personality_data=personality_data
                                )
                                mock_redis_memory.return_value = (
                                    MockRedisMemoryService()
                                )
                                mock_redis_identity.return_value = (
                                    MockRedisIdentityService(
                                        identity_facts=identity_facts
                                    )
                                )
                                mock_redis_context.return_value = (
                                    MockRedisContextService()
                                )

                                # Should accept token budget parameter
                                response = asyncio.run(gentle_refresh(tokens=1000))
                                assert isinstance(response, str)
                                assert len(response) > 0

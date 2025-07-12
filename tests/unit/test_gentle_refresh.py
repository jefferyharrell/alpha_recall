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
            "utc": "2025-01-01T12:00:00.000000+00:00",
            "local": "2025-01-01T05:00:00.000000-07:00",
            "human_readable": "Tuesday, January 01, 2025 05:00 AM",
            "timezone": {
                "name": "America/Los_Angeles",
                "offset": "-07:00",
                "display": "PST",
            },
            "unix_timestamp": 1735732800.0,
            "day_of_week": {"integer": 1, "name": "Tuesday"},
        }


class MockGeolocationService:
    """Mock geolocation service for testing."""

    async def get_location(self):
        """Return mock location."""
        return "Los Angeles"


class MockRedisService:
    """Mock Redis service for testing."""

    def __init__(self, identity_facts=None):
        self.identity_facts = identity_facts or []
        self.client = MockRedisClient()

    def get_identity_facts(self):
        """Return mock identity facts."""
        return self.identity_facts


class MockRedisClient:
    """Mock Redis client for testing."""

    def zrevrange(self, key, start, end, withscores=False):
        """Mock zrevrange - return empty for simplicity."""
        return []

    def hmget(self, key, fields):
        """Mock hmget - return None for simplicity."""
        return [None] * len(fields)


class MockMemgraphService:
    """Mock Memgraph service for testing."""

    def __init__(self, core_identity=None, personality_data=None):
        self.core_identity = core_identity
        self.personality_data = personality_data or []
        self.db = MockDB(personality_data)

    def get_entity_with_observations(self, entity_name):
        """Return mock core identity entity."""
        if self.core_identity:
            return self.core_identity
        return {
            "name": entity_name,
            "observations": [
                {"content": "Mock identity fact 1"},
                {"content": "Mock identity fact 2"},
            ],
        }


class MockDB:
    """Mock database for Memgraph."""

    def __init__(self, personality_data=None):
        self.personality_data = personality_data or []

    def execute_and_fetch(self, query):
        """Mock query execution - return personality data."""
        return self.personality_data


class MockSettings:
    """Mock settings for testing."""

    def __init__(self):
        self.core_identity_node = "Alpha Core Identity"


def test_gentle_refresh_registration():
    """Test that gentle_refresh tools register correctly."""
    mock_mcp = MockMCP()
    register_gentle_refresh_tools(mock_mcp)

    assert "gentle_refresh" in mock_mcp.tools
    assert callable(mock_mcp.tools["gentle_refresh"])


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.GeolocationService", MockGeolocationService)
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_basic_success(mock_redis_service, mock_memgraph_service):
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

    mock_redis_service.return_value = MockRedisService(identity_facts=identity_facts)
    mock_memgraph_service.return_value = MockMemgraphService(
        personality_data=personality_data
    )

    response = asyncio.run(gentle_refresh())

    # Basic success criteria
    assert isinstance(response, str)
    assert len(response) > 100  # Should be substantial prose
    assert response.startswith("Good")
    assert "Los Angeles" in response
    assert "## Core Identity" in response
    assert "## Personality Traits" in response
    assert "## Recent Context" in response


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.GeolocationService", MockGeolocationService)
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_includes_identity_facts(
    mock_redis_service, mock_memgraph_service
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
    mock_redis_service.return_value = MockRedisService(identity_facts=identity_facts)
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
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_includes_personality_traits(
    mock_redis_service, mock_memgraph_service
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
    mock_redis_service.return_value = MockRedisService(identity_facts=identity_facts)
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
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_handles_empty_data(mock_redis_service, mock_memgraph_service):
    """Test graceful handling when all data sources are empty."""
    # Setup with empty data - this should now return initialization error
    mock_redis_service.return_value = MockRedisService(identity_facts=[])
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
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_error_resilience(mock_redis_service, mock_memgraph_service):
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
    mock_redis_service.return_value = MockRedisService(identity_facts=identity_facts)

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
                        "alpha_recall.tools.gentle_refresh.get_redis_service"
                    ) as mock_redis:
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
                        mock_redis.return_value = MockRedisService(
                            identity_facts=identity_facts
                        )

                        # Should not raise exception with token parameter
                        response = asyncio.run(gentle_refresh(tokens=500))
                        assert isinstance(response, str)
                        assert response.startswith("Good")


def test_gentle_refresh_complete_failure():
    """Test behavior when all services fail."""
    # Setup complete failure scenario
    with patch("alpha_recall.tools.gentle_refresh.time_service") as mock_time:
        mock_time.now_async.side_effect = Exception("Time service failed")

        # Should raise exception for complete failure
        try:
            asyncio.run(gentle_refresh())
            raise AssertionError("Expected exception for complete service failure")
        except Exception as e:
            # Exception is expected and acceptable
            assert "failed" in str(e).lower() or "error" in str(e).lower()

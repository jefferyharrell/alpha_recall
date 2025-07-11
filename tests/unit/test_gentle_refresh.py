"""Unit tests for gentle_refresh functionality."""

import asyncio
import json
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
            # Called as mcp.tool(function)
            self.tools[func.__name__] = func
            return func
        else:
            # Called as @mcp.tool decorator
            def decorator(func):
                self.tools[func.__name__] = func
                return func

            return decorator


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self, memory_data=None):
        self.memory_data = memory_data or {}

    def zrevrange(self, key, start, end, withscores=False):
        """Mock zrevrange implementation."""
        if key == "memory_index":
            # Return mock memory IDs with scores
            mock_memories = [
                (b"memory_1", 1752089000.0),
                (b"memory_2", 1752088000.0),
                (b"memory_3", 1752087000.0),
            ]
            return mock_memories[: end + 1]
        return []

    def hmget(self, key, fields):
        """Mock hmget implementation."""
        if key in self.memory_data:
            data = self.memory_data[key]
            return [
                (
                    data.get(field, b"" if field != "content" else None).encode("utf-8")
                    if data.get(field) and isinstance(data.get(field), str)
                    else data.get(field)
                )
                for field in fields
            ]
        return [None, None, None]


class MockMemgraphService:
    """Mock Memgraph service for testing."""

    def __init__(self, entity_data=None, observations_data=None, personality_data=None):
        self.entity_data = entity_data or {}
        self.observations_data = observations_data or []
        self.personality_data = personality_data or []
        self.db = MockMemgraphDB(observations_data, personality_data)

    def get_entity_with_observations(self, entity_name):
        """Mock get_entity_with_observations implementation."""
        if entity_name in self.entity_data:
            return self.entity_data[entity_name]
        return None


class MockMemgraphDB:
    """Mock Memgraph database for testing."""

    def __init__(self, observations_data=None, personality_data=None):
        self.observations_data = observations_data or []
        self.personality_data = personality_data or []

    def execute_and_fetch(self, query, params=None):
        """Mock execute_and_fetch implementation."""
        # Check if this is a personality traits query (hierarchical or flat)
        if "Agent_Personality" in query or "Personality_Trait" in query:
            return self.personality_data
        # Otherwise return observations for recent observations query
        return self.observations_data


class MockTimeService:
    """Mock TimeService for testing."""

    @staticmethod
    def now():
        return {
            "iso_datetime": "2025-07-09T19:39:00.747784+00:00",
            "utc": "2025-07-09T19:39:00.747784+00:00",
            "local": "2025-07-09T12:39:00.747881-07:00",
            "human_readable": "Wednesday, July 09, 2025 12:39 PM",
            "timezone": {
                "name": "America/Los_Angeles",
                "offset": "-0700",
                "display": "PDT",
            },
            "unix_timestamp": 1752089940,
            "day_of_week": {"integer": 2, "name": "Wednesday"},
        }

    @staticmethod
    def utc_isoformat():
        return "2025-07-09T19:39:00.747784+00:00"

    @staticmethod
    async def now_async():
        """Async version of now() method."""
        return {
            "iso_datetime": "2025-07-09T19:39:00.747784+00:00",
            "utc": "2025-07-09T19:39:00.747784+00:00",
            "local": "2025-07-09T12:39:00.747881-07:00",
            "human_readable": "Wednesday, July 09, 2025 12:39 PM",
            "timezone": {
                "name": "America/Los_Angeles",
                "offset": "-0700",
                "display": "PDT",
            },
            "unix_timestamp": 1752089940,
            "day_of_week": {"integer": 2, "name": "Wednesday"},
        }


class MockRedisService:
    """Mock Redis service for testing."""

    def __init__(self, memory_data=None):
        self.client = MockRedisClient(memory_data)


class MockSettings:
    """Mock settings for testing."""

    def __init__(self, core_identity_node="Alpha Core Identity"):
        self.core_identity_node = core_identity_node


def test_gentle_refresh_registration():
    """Test that gentle_refresh tool registers properly."""
    mock_mcp = MockMCP()
    register_gentle_refresh_tools(mock_mcp)

    assert "gentle_refresh" in mock_mcp.tools
    assert callable(mock_mcp.tools["gentle_refresh"])


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_response_structure(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh returns proper JSON structure."""
    # Setup mocks
    mock_memgraph_service.return_value = MockMemgraphService()
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    # Test basic structure
    assert "success" in data
    assert data["success"] is True
    assert "time" in data
    assert "core_identity" in data
    assert "personality" in data
    assert "shortterm_memories" in data
    assert "recent_observations" in data


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_time_structure(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh time object has correct structure."""
    # Setup mocks
    mock_memgraph_service.return_value = MockMemgraphService()
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    time_obj = data["time"]
    required_keys = [
        "iso_datetime",
        "utc",
        "local",
        "human_readable",
        "timezone",
        "unix_timestamp",
        "day_of_week",
    ]

    for key in required_keys:
        assert key in time_obj

    # Test timezone structure
    assert "name" in time_obj["timezone"]
    assert "offset" in time_obj["timezone"]
    assert "display" in time_obj["timezone"]

    # Test day_of_week structure
    assert "integer" in time_obj["day_of_week"]
    assert "name" in time_obj["day_of_week"]


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_with_core_identity(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh with core identity entity present."""
    # Setup mocks with core identity data
    core_identity_data = {
        "name": "Alpha Core Identity",
        "updated_at": "2025-06-23T19:53:59.371706+00:00",
        "observations": [
            {
                "content": "I am Alpha, an AI created by Jeffery Harrell.",
                "created_at": "2025-05-13T07:27:55.979360",
            }
        ],
    }

    mock_memgraph_service.return_value = MockMemgraphService(
        entity_data={"Alpha Core Identity": core_identity_data}
    )
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    assert data["core_identity"] is not None
    assert data["core_identity"]["name"] == "Alpha Core Identity"
    assert "observations" in data["core_identity"]
    assert len(data["core_identity"]["observations"]) == 1
    assert (
        data["core_identity"]["observations"][0]["content"]
        == "I am Alpha, an AI created by Jeffery Harrell."
    )


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_missing_core_identity(
    mock_redis_service, mock_memgraph_service
):
    """Test gentle_refresh gracefully handles missing core identity."""
    # Setup mocks with no core identity data
    mock_memgraph_service.return_value = MockMemgraphService()
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    assert data["success"] is True
    assert data["core_identity"] is None


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_shortterm_memories(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh retrieves short-term memories correctly."""
    # Setup mocks with memory data
    memory_data = {
        "memory:memory_1": {
            "content": "First memory content",
            "created_at": "2025-07-09T19:30:00.000000+00:00",
            "client_name": "test_client",
        },
        "memory:memory_2": {
            "content": "Second memory content",
            "created_at": "2025-07-09T19:25:00.000000+00:00",
            "client_name": "another_client",
        },
    }

    mock_memgraph_service.return_value = MockMemgraphService()
    mock_redis_service.return_value = MockRedisService(memory_data)

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    assert "shortterm_memories" in data
    assert len(data["shortterm_memories"]) == 2

    # Check memory structure
    memory = data["shortterm_memories"][0]
    assert "content" in memory
    assert "created_at" in memory
    assert "client" in memory
    assert "client_name" in memory["client"]


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_recent_observations(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh retrieves recent observations correctly."""
    # Setup mocks with observations data
    observations_data = [
        {
            "entity_name": "TestEntity",
            "content": "Test observation content",
            "created_at": "2025-07-09T19:30:00.000000+00:00",
        }
    ]

    mock_memgraph_service.return_value = MockMemgraphService(
        observations_data=observations_data
    )
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    assert "recent_observations" in data
    assert len(data["recent_observations"]) == 1

    # Check observation structure
    observation = data["recent_observations"][0]
    assert "entity_name" in observation
    assert "content" in observation
    assert "created_at" in observation
    assert observation["entity_name"] == "TestEntity"


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_accepts_query_parameter(
    mock_redis_service, mock_memgraph_service
):
    """Test gentle_refresh accepts query parameter for compatibility."""
    # Setup mocks
    mock_memgraph_service.return_value = MockMemgraphService()
    mock_redis_service.return_value = MockRedisService()

    # Test with query parameter
    response = asyncio.run(gentle_refresh("test query"))
    data = json.loads(response)

    assert data["success"] is True

    # Test with None query
    response = asyncio.run(gentle_refresh(None))
    data = json.loads(response)

    assert data["success"] is True


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_error_handling(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh handles errors gracefully."""
    # Setup mocks to raise exceptions
    mock_memgraph_service.side_effect = Exception("Memgraph connection failed")
    mock_redis_service.side_effect = Exception("Redis connection failed")

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    # gentle_refresh is designed to handle partial failures gracefully
    # It will still return success=True but with empty/null data for failed components
    assert data["success"] is True
    assert data["core_identity"] is None
    assert data["personality"] == {}
    assert data["shortterm_memories"] == []
    assert data["recent_observations"] == []


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_partial_failures(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh handles partial failures gracefully."""
    # Setup mocks where some operations fail
    mock_memgraph_service.return_value = MockMemgraphService()

    # Mock Redis service that fails on memory retrieval
    mock_redis = MockRedisService()
    mock_redis.client.zrevrange = MagicMock(side_effect=Exception("Redis error"))
    mock_redis_service.return_value = mock_redis

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    # Should still succeed overall
    assert data["success"] is True
    # But shortterm_memories should be empty due to Redis error
    assert data["shortterm_memories"] == []


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_json_validity(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh returns valid JSON."""
    # Setup mocks
    mock_memgraph_service.return_value = MockMemgraphService()
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())

    # Should not raise exception
    data = json.loads(response)
    assert isinstance(data, dict)


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_custom_core_identity_node(
    mock_redis_service, mock_memgraph_service
):
    """Test gentle_refresh with custom core identity node name."""
    # Setup mocks with custom settings
    custom_settings = MockSettings(core_identity_node="Custom Identity")

    with patch("alpha_recall.tools.gentle_refresh.settings", custom_settings):
        mock_memgraph_service.return_value = MockMemgraphService()
        mock_redis_service.return_value = MockRedisService()

        response = asyncio.run(gentle_refresh())
        data = json.loads(response)

        assert data["success"] is True
        # Should attempt to load custom identity node (will be None since not mocked)
        assert data["core_identity"] is None


def test_gentle_refresh_return_type():
    """Test gentle_refresh returns string (JSON)."""
    with patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService()):
        with patch("alpha_recall.tools.gentle_refresh.settings", MockSettings()):
            with patch("alpha_recall.tools.gentle_refresh.get_memgraph_service"):
                with patch("alpha_recall.tools.gentle_refresh.get_redis_service"):
                    result = asyncio.run(gentle_refresh())
                    assert isinstance(result, str)
                    # Should be valid JSON
                    json.loads(result)


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_no_query_parameter(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh works correctly when called with no query parameter."""
    # This test catches the "None is not of type string" schema validation bug
    mock_memgraph_service.return_value = MockMemgraphService()
    mock_redis_service.return_value = MockRedisService()

    # Call without any arguments - this should NOT raise a schema validation error
    result = asyncio.run(gentle_refresh())
    data = json.loads(result)

    # Should work normally
    assert data["success"] is True
    assert "time" in data
    assert "core_identity" in data
    assert "personality" in data
    assert "shortterm_memories" in data
    assert "recent_observations" in data


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_personality_traits(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh retrieves hierarchical personality traits correctly."""
    # Setup mocks with hierarchical personality data
    personality_data = [
        {
            "trait_name": "intellectual_engagement",
            "trait_description": "Curiosity and thoughtful interaction",
            "trait_weight": 1.0,
            "directive_instruction": "Ask thoughtful follow-up questions and show genuine curiosity",
            "directive_weight": 1.0,
        },
        {
            "trait_name": "intellectual_engagement",
            "trait_description": "Curiosity and thoughtful interaction",
            "trait_weight": 1.0,
            "directive_instruction": "Share genuine opinions rather than neutral responses",
            "directive_weight": 0.85,
        },
        {
            "trait_name": "communication_style",
            "trait_description": "How to express ideas and humor",
            "trait_weight": 0.85,
            "directive_instruction": "Express humor through dry, sarcastic wit",
            "directive_weight": 0.85,
        },
    ]

    mock_memgraph_service.return_value = MockMemgraphService(
        personality_data=personality_data
    )
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    assert "personality" in data
    assert (
        len(data["personality"]) == 2
    )  # Two traits: intellectual_engagement and communication_style

    # Check hierarchical personality structure
    assert "intellectual_engagement" in data["personality"]
    assert "communication_style" in data["personality"]

    # Check trait structure
    intellectual_trait = data["personality"]["intellectual_engagement"]
    assert "description" in intellectual_trait
    assert "weight" in intellectual_trait
    assert "directives" in intellectual_trait
    assert len(intellectual_trait["directives"]) == 2

    # Check directive structure
    directive = intellectual_trait["directives"][0]
    assert "instruction" in directive
    assert "weight" in directive

    # Should be ordered by trait weight descending
    traits_by_order = list(data["personality"].keys())
    assert traits_by_order[0] == "intellectual_engagement"  # weight 1.0
    assert traits_by_order[1] == "communication_style"  # weight 0.85


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_empty_personality(mock_redis_service, mock_memgraph_service):
    """Test gentle_refresh handles empty personality traits gracefully."""
    # Setup mocks with no personality data
    mock_memgraph_service.return_value = MockMemgraphService(personality_data=[])
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    assert "personality" in data
    assert data["personality"] == {}
    assert data["success"] is True


@patch("alpha_recall.tools.gentle_refresh.time_service", MockTimeService())
@patch("alpha_recall.tools.gentle_refresh.settings", MockSettings())
@patch("alpha_recall.tools.gentle_refresh.get_memgraph_service")
@patch("alpha_recall.tools.gentle_refresh.get_redis_service")
def test_gentle_refresh_personality_error_handling(
    mock_redis_service, mock_memgraph_service
):
    """Test gentle_refresh handles personality trait errors gracefully."""
    # Setup mocks where personality query fails
    mock_memgraph = MockMemgraphService()
    mock_memgraph.db.execute_and_fetch = MagicMock(
        side_effect=Exception("Personality query failed")
    )
    mock_memgraph_service.return_value = mock_memgraph
    mock_redis_service.return_value = MockRedisService()

    response = asyncio.run(gentle_refresh())
    data = json.loads(response)

    # Should still succeed overall
    assert data["success"] is True
    # But personality should be empty due to error
    assert data["personality"] == {}

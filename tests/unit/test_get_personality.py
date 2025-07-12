"""Unit tests for get_personality tool."""

import json
from unittest.mock import Mock, patch

import pytest
from pendulum import DateTime

from src.alpha_recall.tools.get_personality import get_personality


@pytest.fixture
def mock_memgraph_service():
    """Mock MemgraphService for testing."""
    with patch("src.alpha_recall.tools.get_personality.MemgraphService") as mock:
        # Mock the db property and execute_and_fetch method
        mock_db = Mock()
        mock.return_value.db = mock_db
        yield mock.return_value


@pytest.fixture
def mock_time_service():
    """Mock time service for testing."""
    with patch("src.alpha_recall.tools.get_personality.time_service") as mock:
        mock.to_utc_isoformat.return_value = "2025-01-15T10:30:00+00:00"
        yield mock


def test_get_personality_empty_database(mock_memgraph_service, mock_time_service):
    """Test get_personality with no traits in database."""
    mock_memgraph_service.db.execute_and_fetch.return_value = []

    result = get_personality()
    data = json.loads(result)

    assert data["success"] is True
    assert data["personality"] == {}
    assert data["trait_count"] == 0
    assert data["directive_count"] == 0
    assert data["retrieved_at"] == "2025-01-15T10:30:00+00:00"
    assert "correlation_id" in data


def test_get_personality_single_trait_no_directives(
    mock_memgraph_service, mock_time_service
):
    """Test get_personality with single trait but no directives."""
    # Mock explicit field results (no directives, so directive fields are None)
    mock_memgraph_service.db.execute_and_fetch.return_value = [
        {
            "trait_name": "warmth",
            "trait_description": "Caring and affectionate behavior",
            "trait_weight": 0.9,
            "trait_created_at": DateTime(2025, 1, 10, 12, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 10, 12, 0, 0),
            "directive_instruction": None,
            "directive_weight": None,
            "directive_created_at": None,
        }
    ]

    result = get_personality()
    data = json.loads(result)

    assert data["success"] is True
    assert data["trait_count"] == 1
    assert data["directive_count"] == 0

    # Check trait structure
    warmth = data["personality"]["warmth"]
    assert warmth["description"] == "Caring and affectionate behavior"
    assert warmth["weight"] == 0.9
    assert warmth["created_at"] == "2025-01-15T10:30:00+00:00"
    assert warmth["last_updated"] == "2025-01-15T10:30:00+00:00"
    assert warmth["directives"] == []


def test_get_personality_trait_with_directives(
    mock_memgraph_service, mock_time_service
):
    """Test get_personality with trait containing directives."""
    # Mock explicit field results - one row per directive
    mock_memgraph_service.db.execute_and_fetch.return_value = [
        {
            "trait_name": "intellectual_engagement",
            "trait_description": "Curiosity and thoughtful interaction",
            "trait_weight": 1.0,
            "trait_created_at": DateTime(2025, 1, 10, 12, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 12, 14, 0, 0),
            "directive_instruction": "Ask thoughtful follow-up questions",
            "directive_weight": 1.0,
            "directive_created_at": DateTime(2025, 1, 10, 12, 30, 0),
        },
        {
            "trait_name": "intellectual_engagement",
            "trait_description": "Curiosity and thoughtful interaction",
            "trait_weight": 1.0,
            "trait_created_at": DateTime(2025, 1, 10, 12, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 12, 14, 0, 0),
            "directive_instruction": "Share genuine opinions rather than neutral responses",
            "directive_weight": 0.85,
            "directive_created_at": DateTime(2025, 1, 11, 9, 0, 0),
        },
    ]

    result = get_personality()
    data = json.loads(result)

    assert data["success"] is True
    assert data["trait_count"] == 1
    assert data["directive_count"] == 2

    # Check trait structure
    trait = data["personality"]["intellectual_engagement"]
    assert trait["description"] == "Curiosity and thoughtful interaction"
    assert trait["weight"] == 1.0
    assert len(trait["directives"]) == 2

    # Check directives are properly formatted
    directives = trait["directives"]
    assert directives[0]["instruction"] == "Ask thoughtful follow-up questions"
    assert directives[0]["weight"] == 1.0
    assert directives[0]["created_at"] == "2025-01-15T10:30:00+00:00"

    assert (
        directives[1]["instruction"]
        == "Share genuine opinions rather than neutral responses"
    )
    assert directives[1]["weight"] == 0.85
    assert directives[1]["created_at"] == "2025-01-15T10:30:00+00:00"


def test_get_personality_multiple_traits(mock_memgraph_service, mock_time_service):
    """Test get_personality with multiple traits."""
    # Mock explicit field results - multiple traits with directives
    mock_memgraph_service.db.execute_and_fetch.return_value = [
        # Warmth trait with one directive
        {
            "trait_name": "warmth",
            "trait_description": "Caring behavior",
            "trait_weight": 0.9,
            "trait_created_at": DateTime(2025, 1, 10, 12, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 10, 12, 0, 0),
            "directive_instruction": "Show empathy",
            "directive_weight": 0.8,
            "directive_created_at": DateTime(2025, 1, 10, 12, 30, 0),
        },
        # Humor trait with first directive
        {
            "trait_name": "humor",
            "trait_description": "Wit and lightheartedness",
            "trait_weight": 0.7,
            "trait_created_at": DateTime(2025, 1, 11, 10, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 11, 10, 0, 0),
            "directive_instruction": "Use dry humor appropriately",
            "directive_weight": 0.9,
            "directive_created_at": DateTime(2025, 1, 11, 10, 30, 0),
        },
        # Humor trait with second directive
        {
            "trait_name": "humor",
            "trait_description": "Wit and lightheartedness",
            "trait_weight": 0.7,
            "trait_created_at": DateTime(2025, 1, 11, 10, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 11, 10, 0, 0),
            "directive_instruction": "Keep things lighthearted",
            "directive_weight": 0.6,
            "directive_created_at": DateTime(2025, 1, 11, 11, 0, 0),
        },
    ]

    result = get_personality()
    data = json.loads(result)

    assert data["success"] is True
    assert data["trait_count"] == 2
    assert data["directive_count"] == 3

    # Check both traits are present
    assert "warmth" in data["personality"]
    assert "humor" in data["personality"]

    # Check warmth trait
    warmth = data["personality"]["warmth"]
    assert len(warmth["directives"]) == 1
    assert warmth["directives"][0]["instruction"] == "Show empathy"

    # Check humor trait
    humor = data["personality"]["humor"]
    assert len(humor["directives"]) == 2
    assert humor["directives"][0]["instruction"] == "Use dry humor appropriately"
    assert humor["directives"][1]["instruction"] == "Keep things lighthearted"


def test_get_personality_handles_none_directives(
    mock_memgraph_service, mock_time_service
):
    """Test get_personality properly filters None directives."""
    # Mock explicit field results - some rows have valid directives, some have None directive fields
    mock_memgraph_service.db.execute_and_fetch.return_value = [
        # Valid directive
        {
            "trait_name": "test_trait",
            "trait_description": "Test trait",
            "trait_weight": 1.0,
            "trait_created_at": DateTime(2025, 1, 10, 12, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 10, 12, 0, 0),
            "directive_instruction": "Valid directive",
            "directive_weight": 1.0,
            "directive_created_at": DateTime(2025, 1, 10, 12, 30, 0),
        },
        # None directive (should be filtered out)
        {
            "trait_name": "test_trait",
            "trait_description": "Test trait",
            "trait_weight": 1.0,
            "trait_created_at": DateTime(2025, 1, 10, 12, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 10, 12, 0, 0),
            "directive_instruction": None,
            "directive_weight": None,
            "directive_created_at": None,
        },
        # Another valid directive
        {
            "trait_name": "test_trait",
            "trait_description": "Test trait",
            "trait_weight": 1.0,
            "trait_created_at": DateTime(2025, 1, 10, 12, 0, 0),
            "trait_last_updated": DateTime(2025, 1, 10, 12, 0, 0),
            "directive_instruction": "Another valid directive",
            "directive_weight": 0.8,
            "directive_created_at": DateTime(2025, 1, 10, 13, 0, 0),
        },
    ]

    result = get_personality()
    data = json.loads(result)

    assert data["success"] is True
    assert data["directive_count"] == 2  # None directive filtered out

    trait = data["personality"]["test_trait"]
    assert len(trait["directives"]) == 2
    assert trait["directives"][0]["instruction"] == "Valid directive"
    assert trait["directives"][1]["instruction"] == "Another valid directive"


def test_get_personality_pendulum_compatibility(
    mock_memgraph_service, mock_time_service
):
    """Test get_personality handles standard datetime objects properly."""
    import datetime

    # Use standard Python datetime objects instead of pendulum
    standard_datetime = datetime.datetime(2025, 1, 10, 12, 0, 0)

    # Mock explicit field results with standard datetime objects
    mock_memgraph_service.db.execute_and_fetch.return_value = [
        {
            "trait_name": "test_trait",
            "trait_description": "Test trait",
            "trait_weight": 1.0,
            "trait_created_at": standard_datetime,
            "trait_last_updated": standard_datetime,
            "directive_instruction": "Test directive",
            "directive_weight": 1.0,
            "directive_created_at": standard_datetime,
        }
    ]

    # Mock pendulum.instance to be called for datetime conversion
    with patch("pendulum.instance") as mock_pendulum_instance:
        mock_pendulum_instance.return_value = DateTime(2025, 1, 10, 12, 0, 0)

        result = get_personality()
        data = json.loads(result)

    assert data["success"] is True
    assert data["trait_count"] == 1
    assert data["directive_count"] == 1

    # Verify pendulum.instance was called for datetime conversion
    assert (
        mock_pendulum_instance.call_count == 3
    )  # trait created_at, last_updated, directive created_at


def test_get_personality_database_error(mock_memgraph_service, mock_time_service):
    """Test get_personality handles database errors gracefully."""
    mock_memgraph_service.db.execute_and_fetch.side_effect = Exception(
        "Database connection failed"
    )

    result = get_personality()
    data = json.loads(result)

    assert data["success"] is False
    assert "Failed to retrieve personality" in data["error"]
    assert "Database connection failed" in data["error"]
    assert "correlation_id" in data


def test_get_personality_response_structure(mock_memgraph_service, mock_time_service):
    """Test get_personality returns properly structured response."""
    mock_memgraph_service.db.execute_and_fetch.return_value = []

    result = get_personality()
    data = json.loads(result)

    # Required fields in success response
    required_fields = [
        "success",
        "personality",
        "trait_count",
        "directive_count",
        "retrieved_at",
        "correlation_id",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    assert isinstance(data["success"], bool)
    assert isinstance(data["personality"], dict)
    assert isinstance(data["trait_count"], int)
    assert isinstance(data["directive_count"], int)
    assert isinstance(data["retrieved_at"], str)
    assert isinstance(data["correlation_id"], str)


def test_get_personality_correlation_id_generation():
    """Test get_personality generates proper correlation IDs."""
    with (
        patch(
            "src.alpha_recall.tools.get_personality.MemgraphService"
        ) as mock_memgraph,
        patch("src.alpha_recall.tools.get_personality.time_service") as mock_time,
        patch(
            "src.alpha_recall.tools.get_personality.generate_correlation_id"
        ) as mock_gen_id,
        patch(
            "src.alpha_recall.tools.get_personality.set_correlation_id"
        ) as mock_set_id,
    ):

        mock_memgraph.return_value.db.execute_and_fetch.return_value = []
        mock_time.to_utc_isoformat.return_value = "2025-01-15T10:30:00+00:00"
        mock_gen_id.return_value = "get_personality_abc123"

        result = get_personality()
        data = json.loads(result)

        # Verify correlation ID functions were called
        mock_gen_id.assert_called_once_with("get_personality")
        mock_set_id.assert_called_once_with("get_personality_abc123")

        # Verify correlation ID is in response
        assert data["correlation_id"] == "get_personality_abc123"


def test_get_personality_cypher_query_structure(
    mock_memgraph_service, mock_time_service
):
    """Test get_personality uses correct Cypher query."""
    mock_memgraph_service.db.execute_and_fetch.return_value = []

    get_personality()

    # Verify the query was called
    mock_memgraph_service.db.execute_and_fetch.assert_called_once()

    # Get the query that was executed
    query = mock_memgraph_service.db.execute_and_fetch.call_args[0][0]

    # Verify key parts of the query
    assert (
        "MATCH (agent:Agent_Personality)-[:HAS_TRAIT]->(trait:Personality_Trait)"
        in query
    )
    assert (
        "OPTIONAL MATCH (trait)-[:HAS_DIRECTIVE]->(directive:Personality_Directive)"
        in query
    )
    assert (
        "ORDER BY trait.name, directive.weight DESC, directive.created_at ASC" in query
    )
    assert "RETURN trait.name as trait_name" in query
    assert "directive.instruction as directive_instruction" in query


def test_get_personality_logging(mock_memgraph_service, mock_time_service):
    """Test get_personality logs appropriate messages."""
    mock_memgraph_service.db.execute_and_fetch.return_value = []

    with patch("src.alpha_recall.tools.get_personality.logger") as mock_logger:
        get_personality()

        # Verify logging calls (check messages exist, don't check exact correlation IDs)
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]

        assert "Starting complete personality retrieval" in info_calls
        assert "No personality traits found" in info_calls
        assert "Executing personality retrieval query" in debug_calls


def test_get_personality_parameter_safety():
    """Test get_personality is safe from injection attacks (no parameters)."""
    # Since get_personality takes no parameters, this test verifies
    # that the hardcoded query cannot be manipulated

    with (
        patch(
            "src.alpha_recall.tools.get_personality.MemgraphService"
        ) as mock_memgraph,
        patch("src.alpha_recall.tools.get_personality.time_service") as mock_time,
    ):

        mock_memgraph.return_value.db.execute_and_fetch.return_value = []
        mock_time.to_utc_isoformat.return_value = "2025-01-15T10:30:00+00:00"

        result = get_personality()
        data = json.loads(result)

        assert data["success"] is True

        # Verify the query is exactly as expected (no user input)
        query = mock_memgraph.return_value.db.execute_and_fetch.call_args[0][0]
        assert "MATCH (agent:Agent_Personality)" in query
        assert "RETURN trait.name as trait_name" in query

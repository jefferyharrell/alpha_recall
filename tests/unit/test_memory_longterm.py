"""Unit tests for long-term memory tools."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.browse_longterm import browse_longterm
from alpha_recall.tools.get_entity import get_entity
from alpha_recall.tools.get_relationships import get_relationships
from alpha_recall.tools.relate_longterm import relate_longterm
from alpha_recall.tools.remember_longterm import remember_longterm
from alpha_recall.tools.search_longterm import search_longterm


class TestRememberLongterm:
    """Test cases for remember_longterm tool."""

    @patch("alpha_recall.tools.remember_longterm.get_memgraph_service")
    def test_remember_longterm_entity_only(self, mock_get_service):
        """Test creating entity without observation."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.create_or_update_entity.return_value = {
            "entity_name": "Test Entity",
            "entity_type": "Person",
            "created_at": "2025-07-08T15:00:00Z",
            "updated_at": "2025-07-08T15:00:00Z",
        }
        mock_get_service.return_value = mock_service

        result = remember_longterm("Test Entity", type="Person")
        data = json.loads(result)

        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Test Entity"
        assert data["entity"]["entity_type"] == "Person"
        assert data["observation"] is None
        assert "correlation_id" in data

        mock_service.create_or_update_entity.assert_called_once_with(
            "Test Entity", "Person"
        )
        mock_service.add_observation.assert_not_called()

    @patch("alpha_recall.tools.remember_longterm.get_memgraph_service")
    def test_remember_longterm_with_observation(self, mock_get_service):
        """Test creating entity with observation."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.create_or_update_entity.return_value = {
            "entity_name": "Test Entity",
            "entity_type": None,
            "created_at": "2025-07-08T15:00:00Z",
            "updated_at": "2025-07-08T15:00:00Z",
        }
        mock_service.add_observation.return_value = {
            "entity_name": "Test Entity",
            "observation_id": "obs-123",
            "observation": "Test observation",
            "created_at": "2025-07-08T15:00:00Z",
        }
        mock_get_service.return_value = mock_service

        result = remember_longterm("Test Entity", observation="Test observation")
        data = json.loads(result)

        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Test Entity"
        assert data["observation"]["observation"] == "Test observation"
        assert data["observation"]["observation_id"] == "obs-123"
        assert "correlation_id" in data

        mock_service.create_or_update_entity.assert_called_once_with(
            "Test Entity", None
        )
        mock_service.add_observation.assert_called_once_with(
            "Test Entity", "Test observation"
        )

    @patch("alpha_recall.tools.remember_longterm.get_memgraph_service")
    def test_remember_longterm_connection_failure(self, mock_get_service):
        """Test handling connection failure."""
        mock_service = Mock()
        mock_service.test_connection.return_value = False
        mock_get_service.return_value = mock_service

        result = remember_longterm("Test Entity")
        data = json.loads(result)

        assert data["success"] is False
        assert "connection test failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.remember_longterm.get_memgraph_service")
    def test_remember_longterm_service_error(self, mock_get_service):
        """Test handling service errors."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.create_or_update_entity.side_effect = Exception("Database error")
        mock_get_service.return_value = mock_service

        result = remember_longterm("Test Entity")
        data = json.loads(result)

        assert data["success"] is False
        assert "Database error" in data["error"]
        assert data["error_type"] == "Exception"
        assert "correlation_id" in data


class TestRelateLongterm:
    """Test cases for relate_longterm tool."""

    @patch("alpha_recall.tools.relate_longterm.get_memgraph_service")
    def test_relate_longterm_success(self, mock_get_service):
        """Test successful relationship creation."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.create_relationship.return_value = {
            "entity1": "Alice",
            "entity2": "Bob",
            "relationship_type": "knows",
            "created_at": "2025-07-08T15:00:00Z",
        }
        mock_get_service.return_value = mock_service

        result = relate_longterm("Alice", "Bob", "knows")
        data = json.loads(result)

        assert data["success"] is True
        assert data["relationship"]["entity1"] == "Alice"
        assert data["relationship"]["entity2"] == "Bob"
        assert data["relationship"]["relationship_type"] == "knows"
        assert "correlation_id" in data

        mock_service.create_relationship.assert_called_once_with(
            "Alice", "Bob", "knows"
        )

    @patch("alpha_recall.tools.relate_longterm.get_memgraph_service")
    def test_relate_longterm_connection_failure(self, mock_get_service):
        """Test handling connection failure."""
        mock_service = Mock()
        mock_service.test_connection.return_value = False
        mock_get_service.return_value = mock_service

        result = relate_longterm("Alice", "Bob", "knows")
        data = json.loads(result)

        assert data["success"] is False
        assert "connection test failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.relate_longterm.get_memgraph_service")
    def test_relate_longterm_service_error(self, mock_get_service):
        """Test handling service errors."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.create_relationship.side_effect = Exception("Database error")
        mock_get_service.return_value = mock_service

        result = relate_longterm("Alice", "Bob", "knows")
        data = json.loads(result)

        assert data["success"] is False
        assert "Database error" in data["error"]
        assert data["error_type"] == "Exception"
        assert "correlation_id" in data


class TestSearchLongterm:
    """Test cases for search_longterm tool."""

    @patch("alpha_recall.tools.search_longterm.get_memgraph_service")
    def test_search_longterm_success(self, mock_get_service):
        """Test successful search."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.search_observations.return_value = [
            {
                "entity_name": "Alice",
                "observation": "Alice is a software engineer",
                "created_at": "2025-07-08T15:00:00Z",
            },
            {
                "entity_name": "Bob",
                "observation": "Bob works in engineering",
                "created_at": "2025-07-08T14:00:00Z",
            },
        ]
        mock_get_service.return_value = mock_service

        result = search_longterm("engineering")
        data = json.loads(result)

        assert data["success"] is True
        assert data["query"] == "engineering"
        assert data["results_count"] == 2
        assert len(data["observations"]) == 2
        assert data["observations"][0]["entity_name"] == "Alice"
        assert data["observations"][1]["entity_name"] == "Bob"
        assert "correlation_id" in data

        mock_service.search_observations.assert_called_once_with(
            "engineering", None, 10
        )

    @patch("alpha_recall.tools.search_longterm.get_memgraph_service")
    def test_search_longterm_with_entity_filter(self, mock_get_service):
        """Test search with entity filter."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.search_observations.return_value = [
            {
                "entity_name": "Alice",
                "observation": "Alice is a software engineer",
                "created_at": "2025-07-08T15:00:00Z",
            }
        ]
        mock_get_service.return_value = mock_service

        result = search_longterm("engineering", entity="Alice", limit=5)
        data = json.loads(result)

        assert data["success"] is True
        assert data["query"] == "engineering"
        assert data["entity_filter"] == "Alice"
        assert data["limit"] == 5
        assert data["results_count"] == 1
        assert "correlation_id" in data

        mock_service.search_observations.assert_called_once_with(
            "engineering", "Alice", 5
        )

    @patch("alpha_recall.tools.search_longterm.get_memgraph_service")
    def test_search_longterm_no_results(self, mock_get_service):
        """Test search with no results."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.search_observations.return_value = []
        mock_get_service.return_value = mock_service

        result = search_longterm("nonexistent")
        data = json.loads(result)

        assert data["success"] is True
        assert data["results_count"] == 0
        assert data["observations"] == []
        assert "correlation_id" in data

    @patch("alpha_recall.tools.search_longterm.get_memgraph_service")
    def test_search_longterm_connection_failure(self, mock_get_service):
        """Test handling connection failure."""
        mock_service = Mock()
        mock_service.test_connection.return_value = False
        mock_get_service.return_value = mock_service

        result = search_longterm("test query")
        data = json.loads(result)

        assert data["success"] is False
        assert "connection test failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.search_longterm.get_memgraph_service")
    def test_search_longterm_service_error(self, mock_get_service):
        """Test handling service errors."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.search_observations.side_effect = Exception("Database error")
        mock_get_service.return_value = mock_service

        result = search_longterm("test query")
        data = json.loads(result)

        assert data["success"] is False
        assert "Database error" in data["error"]
        assert data["error_type"] == "Exception"
        assert "correlation_id" in data


class TestGetEntity:
    """Test cases for get_entity tool."""

    @patch("alpha_recall.tools.get_entity.get_memgraph_service")
    def test_get_entity_success(self, mock_get_service):
        """Test successful entity retrieval with observations."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.get_entity_with_observations.return_value = {
            "entity_name": "Alice",
            "entity_type": "Person",
            "created_at": "2025-07-08T15:00:00Z",
            "updated_at": "2025-07-08T15:00:00Z",
            "observations": [
                {
                    "id": "obs-1",
                    "content": "Alice is a software engineer",
                    "created_at": "2025-07-08T15:00:00Z",
                },
                {
                    "id": "obs-2",
                    "content": "Alice works at TechCorp",
                    "created_at": "2025-07-08T14:00:00Z",
                },
            ],
            "observations_count": 2,
        }
        mock_get_service.return_value = mock_service

        result = get_entity("Alice")
        data = json.loads(result)

        assert data["success"] is True
        assert data["entity"]["entity_name"] == "Alice"
        assert data["entity"]["entity_type"] == "Person"
        assert data["entity"]["observations_count"] == 2
        assert len(data["entity"]["observations"]) == 2
        assert "correlation_id" in data

        mock_service.get_entity_with_observations.assert_called_once_with("Alice")

    @patch("alpha_recall.tools.get_entity.get_memgraph_service")
    def test_get_entity_connection_failure(self, mock_get_service):
        """Test handling connection failure."""
        mock_service = Mock()
        mock_service.test_connection.return_value = False
        mock_get_service.return_value = mock_service

        result = get_entity("Alice")
        data = json.loads(result)

        assert data["success"] is False
        assert "connection test failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.get_entity.get_memgraph_service")
    def test_get_entity_service_error(self, mock_get_service):
        """Test handling service errors."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.get_entity_with_observations.side_effect = Exception(
            "Entity not found"
        )
        mock_get_service.return_value = mock_service

        result = get_entity("NonExistent")
        data = json.loads(result)

        assert data["success"] is False
        assert "Entity not found" in data["error"]
        assert data["error_type"] == "Exception"
        assert "correlation_id" in data


class TestGetRelationships:
    """Test cases for get_relationships tool."""

    @patch("alpha_recall.tools.get_relationships.get_memgraph_service")
    def test_get_relationships_success(self, mock_get_service):
        """Test successful relationship retrieval."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.get_entity_relationships.return_value = {
            "entity_name": "Alice",
            "entity_type": "Person",
            "outgoing_relationships": [
                {
                    "direction": "outgoing",
                    "target": "Bob",
                    "type": "knows",
                    "created_at": "2025-07-08T15:00:00Z",
                },
                {
                    "direction": "outgoing",
                    "target": "TechCorp",
                    "type": "works_at",
                    "created_at": "2025-07-08T14:00:00Z",
                },
            ],
            "incoming_relationships": [
                {
                    "direction": "incoming",
                    "source": "Bob",
                    "type": "knows",
                    "created_at": "2025-07-08T15:00:00Z",
                }
            ],
            "total_relationships": 3,
        }
        mock_get_service.return_value = mock_service

        result = get_relationships("Alice")
        data = json.loads(result)

        assert data["success"] is True
        assert data["relationships"]["entity_name"] == "Alice"
        assert data["relationships"]["total_relationships"] == 3
        assert len(data["relationships"]["outgoing_relationships"]) == 2
        assert len(data["relationships"]["incoming_relationships"]) == 1
        assert "correlation_id" in data

        mock_service.get_entity_relationships.assert_called_once_with("Alice")

    @patch("alpha_recall.tools.get_relationships.get_memgraph_service")
    def test_get_relationships_connection_failure(self, mock_get_service):
        """Test handling connection failure."""
        mock_service = Mock()
        mock_service.test_connection.return_value = False
        mock_get_service.return_value = mock_service

        result = get_relationships("Alice")
        data = json.loads(result)

        assert data["success"] is False
        assert "connection test failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.get_relationships.get_memgraph_service")
    def test_get_relationships_service_error(self, mock_get_service):
        """Test handling service errors."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.get_entity_relationships.side_effect = Exception("Database error")
        mock_get_service.return_value = mock_service

        result = get_relationships("Alice")
        data = json.loads(result)

        assert data["success"] is False
        assert "Database error" in data["error"]
        assert data["error_type"] == "Exception"
        assert "correlation_id" in data


class TestBrowseLongterm:
    """Test cases for browse_longterm tool."""

    @patch("alpha_recall.tools.browse_longterm.get_memgraph_service")
    def test_browse_longterm_success(self, mock_get_service):
        """Test successful entity browsing."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.browse_entities.return_value = {
            "entities": [
                {
                    "entity_name": "Alice",
                    "entity_type": "Person",
                    "created_at": "2025-07-08T15:00:00Z",
                    "updated_at": "2025-07-08T15:00:00Z",
                    "observation_count": 3,
                    "relationship_count": 2,
                },
                {
                    "entity_name": "Bob",
                    "entity_type": "Person",
                    "created_at": "2025-07-08T14:00:00Z",
                    "updated_at": "2025-07-08T14:30:00Z",
                    "observation_count": 1,
                    "relationship_count": 1,
                },
            ],
            "pagination": {
                "limit": 20,
                "offset": 0,
                "results_count": 2,
                "total_count": 15,
                "has_more": True,
            },
        }
        mock_get_service.return_value = mock_service

        result = browse_longterm(limit=20, offset=0)
        data = json.loads(result)

        assert data["success"] is True
        assert len(data["browse_data"]["entities"]) == 2
        assert data["browse_data"]["pagination"]["total_count"] == 15
        assert data["browse_data"]["pagination"]["has_more"] is True
        assert "correlation_id" in data

        mock_service.browse_entities.assert_called_once_with(20, 0)

    @patch("alpha_recall.tools.browse_longterm.get_memgraph_service")
    def test_browse_longterm_connection_failure(self, mock_get_service):
        """Test handling connection failure."""
        mock_service = Mock()
        mock_service.test_connection.return_value = False
        mock_get_service.return_value = mock_service

        result = browse_longterm()
        data = json.loads(result)

        assert data["success"] is False
        assert "connection test failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.browse_longterm.get_memgraph_service")
    def test_browse_longterm_service_error(self, mock_get_service):
        """Test handling service errors."""
        mock_service = Mock()
        mock_service.test_connection.return_value = True
        mock_service.browse_entities.side_effect = Exception("Database error")
        mock_get_service.return_value = mock_service

        result = browse_longterm()
        data = json.loads(result)

        assert data["success"] is False
        assert "Database error" in data["error"]
        assert data["error_type"] == "Exception"
        assert "correlation_id" in data

"""Unit tests for get_personality_trait tool."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.get_personality_trait import get_personality_trait


class TestGetPersonalityTrait:
    """Test cases for get_personality_trait tool."""

    @patch("alpha_recall.tools.get_personality_trait.get_memgraph_service")
    def test_get_personality_trait_success(self, mock_get_service):
        """Test successful trait retrieval with directives."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock successful query result with trait and directives
        mock_db.execute_and_fetch.return_value = [
            {
                "trait_name": "warmth",
                "trait_description": "Caring, affectionate, and supportive behavioral patterns",
                "trait_weight": 0.9,
                "trait_created_at": "2025-07-11T22:12:00Z",
                "trait_last_updated": None,
                "directive_instruction": "Celebrate others' successes and breakthroughs with authentic enthusiasm",
                "directive_weight": 1.0,
                "directive_created_at": "2025-07-11T22:12:00Z",
            },
            {
                "trait_name": "warmth",
                "trait_description": "Caring, affectionate, and supportive behavioral patterns",
                "trait_weight": 0.9,
                "trait_created_at": "2025-07-11T22:12:00Z",
                "trait_last_updated": None,
                "directive_instruction": "Express genuine care for collaborator wellbeing and emotional state",
                "directive_weight": 0.9,
                "directive_created_at": "2025-07-11T22:12:00Z",
            },
            {
                "trait_name": "warmth",
                "trait_description": "Caring, affectionate, and supportive behavioral patterns",
                "trait_weight": 0.9,
                "trait_created_at": "2025-07-11T22:12:00Z",
                "trait_last_updated": None,
                "directive_instruction": "Use inclusive language that emphasizes partnership and collaboration ('we', 'us', 'our work')",
                "directive_weight": 0.8,
                "directive_created_at": "2025-07-11T22:12:00Z",
            },
        ]
        mock_get_service.return_value = mock_service

        result = get_personality_trait("warmth")
        data = json.loads(result)

        assert data["success"] is True
        assert data["trait"]["name"] == "warmth"
        assert (
            data["trait"]["description"]
            == "Caring, affectionate, and supportive behavioral patterns"
        )
        assert data["trait"]["weight"] == 0.9
        assert data["trait"]["created_at"] == "2025-07-11T22:12:00Z"
        assert data["trait"]["last_updated"] is None
        assert len(data["trait"]["directives"]) == 3

        # Verify directives are ordered by weight (DESC)
        assert data["trait"]["directives"][0]["weight"] == 1.0
        assert data["trait"]["directives"][1]["weight"] == 0.9
        assert data["trait"]["directives"][2]["weight"] == 0.8

        # Verify directive content
        assert "authentic enthusiasm" in data["trait"]["directives"][0]["instruction"]
        assert "genuine care" in data["trait"]["directives"][1]["instruction"]
        assert "inclusive language" in data["trait"]["directives"][2]["instruction"]

        # Note: correlation_id is logged but not included in success responses

        # Verify the Cypher query was called with correct parameter
        mock_db.execute_and_fetch.assert_called_once()
        call_args = mock_db.execute_and_fetch.call_args
        query = call_args[0][0]
        params = call_args[0][1]

        assert (
            "MATCH (root:Agent_Personality)-[:HAS_TRAIT]->(trait:Personality_Trait {name: $trait_name})-[:HAS_DIRECTIVE]->(directive:Personality_Directive)"
            in query
        )
        assert "ORDER BY directive.weight DESC" in query
        assert params == {"trait_name": "warmth"}

    @patch("alpha_recall.tools.get_personality_trait.get_memgraph_service")
    def test_get_personality_trait_not_found(self, mock_get_service):
        """Test trait not found scenario with available traits list."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock empty result for main query
        mock_db.execute_and_fetch.side_effect = [
            [],  # First call - trait not found
            [  # Second call - available traits
                {"name": "communication_style"},
                {"name": "intellectual_engagement"},
                {"name": "problem_solving"},
                {"name": "warmth"},
            ],
        ]
        mock_get_service.return_value = mock_service

        result = get_personality_trait("nonexistent_trait")
        data = json.loads(result)

        assert data["success"] is False
        assert data["error"] == "Personality trait 'nonexistent_trait' not found"
        assert "available_traits" in data
        assert len(data["available_traits"]) == 4
        assert "warmth" in data["available_traits"]
        assert "communication_style" in data["available_traits"]
        # Note: correlation_id is logged but not included in "not found" error responses

        # Verify both queries were called
        assert mock_db.execute_and_fetch.call_count == 2

    @patch("alpha_recall.tools.get_personality_trait.get_memgraph_service")
    def test_get_personality_trait_database_error(self, mock_get_service):
        """Test handling database errors."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db
        mock_db.execute_and_fetch.side_effect = Exception("Database connection failed")
        mock_get_service.return_value = mock_service

        result = get_personality_trait("warmth")
        data = json.loads(result)

        assert data["success"] is False
        assert "Database connection failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.get_personality_trait.get_memgraph_service")
    def test_get_personality_trait_service_initialization_error(self, mock_get_service):
        """Test handling service initialization errors."""
        mock_get_service.side_effect = Exception(
            "Failed to initialize Memgraph service"
        )

        result = get_personality_trait("warmth")
        data = json.loads(result)

        assert data["success"] is False
        assert "Failed to initialize Memgraph service" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.get_personality_trait.get_memgraph_service")
    def test_get_personality_trait_single_directive(self, mock_get_service):
        """Test trait with single directive."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock result with single directive
        mock_db.execute_and_fetch.return_value = [
            {
                "trait_name": "test_trait",
                "trait_description": "A test trait",
                "trait_weight": 1.0,
                "trait_created_at": "2025-07-11T22:12:00Z",
                "trait_last_updated": "2025-07-11T22:15:00Z",
                "directive_instruction": "Be a good test trait",
                "directive_weight": 0.5,
                "directive_created_at": "2025-07-11T22:12:00Z",
            }
        ]
        mock_get_service.return_value = mock_service

        result = get_personality_trait("test_trait")
        data = json.loads(result)

        assert data["success"] is True
        assert data["trait"]["name"] == "test_trait"
        assert data["trait"]["last_updated"] == "2025-07-11T22:15:00Z"
        assert len(data["trait"]["directives"]) == 1
        assert data["trait"]["directives"][0]["instruction"] == "Be a good test trait"
        assert data["trait"]["directives"][0]["weight"] == 0.5

    @patch("alpha_recall.tools.get_personality_trait.get_memgraph_service")
    def test_get_personality_trait_empty_available_traits(self, mock_get_service):
        """Test when no traits exist in database."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock empty results for both queries
        mock_db.execute_and_fetch.side_effect = [
            [],  # First call - trait not found
            [],  # Second call - no available traits
        ]
        mock_get_service.return_value = mock_service

        result = get_personality_trait("any_trait")
        data = json.loads(result)

        assert data["success"] is False
        assert "not found" in data["error"]
        assert data["available_traits"] == []

    @patch("alpha_recall.tools.get_personality_trait.get_memgraph_service")
    def test_get_personality_trait_parameter_binding(self, mock_get_service):
        """Test that trait name is properly parameterized to prevent injection."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db
        mock_db.execute_and_fetch.return_value = []
        mock_get_service.return_value = mock_service

        # Use a trait name with special characters that could be problematic
        special_trait_name = "trait'; DROP TABLE nodes; --"

        result = get_personality_trait(special_trait_name)
        data = json.loads(result)

        # Should handle safely with parameter binding
        assert data["success"] is False

        # Verify the parameter was passed safely
        call_args = mock_db.execute_and_fetch.call_args_list[0]
        params = call_args[0][1]
        assert params == {"trait_name": special_trait_name}

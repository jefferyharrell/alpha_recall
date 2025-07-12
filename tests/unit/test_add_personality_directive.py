"""Unit tests for add_personality_directive tool."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.add_personality_directive import add_personality_directive


class TestAddPersonalityDirective:
    """Test cases for add_personality_directive tool."""

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_success(self, mock_get_service):
        """Test successful directive addition to existing trait."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait exists check
        mock_db.execute_and_fetch.side_effect = [
            [
                {
                    "trait_name": "warmth",
                    "trait_description": "Caring behavioral patterns",
                }
            ],  # Trait check
            [],  # Duplicate check (no duplicates)
            [
                {  # Successful creation
                    "directive_instruction": "Be empathetic in conversations",
                    "directive_weight": 0.8,
                    "directive_created_at": "2025-07-11T16:00:00+00:00",
                    "trait_name": "warmth",
                    "trait_description": "Caring behavioral patterns",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = add_personality_directive(
            "warmth", "Be empathetic in conversations", 0.8
        )
        data = json.loads(result)

        assert data["success"] is True
        assert data["trait_name"] == "warmth"
        assert data["trait_description"] == "Caring behavioral patterns"
        assert (
            data["directive_added"]["instruction"] == "Be empathetic in conversations"
        )
        assert data["directive_added"]["weight"] == 0.8
        assert "created_at" in data["directive_added"]

        # Verify all queries were called
        assert mock_db.execute_and_fetch.call_count == 3

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_trait_not_found(self, mock_get_service):
        """Test adding directive to nonexistent trait."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait doesn't exist
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check - empty result
            [  # Available traits
                {"name": "communication_style"},
                {"name": "intellectual_engagement"},
                {"name": "warmth"},
            ],
        ]
        mock_get_service.return_value = mock_service

        result = add_personality_directive("nonexistent", "Some instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "not found" in data["error"]
        assert "available_traits" in data
        assert len(data["available_traits"]) == 3
        assert "warmth" in data["available_traits"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_duplicate_detection(self, mock_get_service):
        """Test detection of duplicate directives."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait exists and directive already exists
        mock_db.execute_and_fetch.side_effect = [
            [
                {
                    "trait_name": "warmth",
                    "trait_description": "Caring behavioral patterns",
                }
            ],  # Trait check
            [{"instruction": "Be empathetic", "weight": 0.9}],  # Duplicate found
        ]
        mock_get_service.return_value = mock_service

        result = add_personality_directive("warmth", "Be empathetic", 0.8)
        data = json.loads(result)

        assert data["success"] is False
        assert "already exists" in data["error"]
        assert "existing_directive" in data
        assert data["existing_directive"]["instruction"] == "Be empathetic"
        assert data["existing_directive"]["weight"] == 0.9
        assert "correlation_id" in data

        # Should only call trait check and duplicate check
        assert mock_db.execute_and_fetch.call_count == 2

    def test_add_personality_directive_validation_empty_trait_name(self):
        """Test validation for empty trait name."""
        result = add_personality_directive("", "Some instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Trait name cannot be empty" in data["error"]
        assert "correlation_id" in data

    def test_add_personality_directive_validation_whitespace_trait_name(self):
        """Test validation for whitespace-only trait name."""
        result = add_personality_directive("   ", "Some instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Trait name cannot be empty" in data["error"]

    def test_add_personality_directive_validation_empty_instruction(self):
        """Test validation for empty instruction."""
        result = add_personality_directive("warmth", "", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Instruction cannot be empty" in data["error"]
        assert "correlation_id" in data

    def test_add_personality_directive_validation_whitespace_instruction(self):
        """Test validation for whitespace-only instruction."""
        result = add_personality_directive("warmth", "   \n\t   ", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Instruction cannot be empty" in data["error"]

    def test_add_personality_directive_validation_invalid_weight_negative(self):
        """Test validation for negative weight."""
        result = add_personality_directive("warmth", "Valid instruction", -0.1)
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]
        assert "correlation_id" in data

    def test_add_personality_directive_validation_invalid_weight_too_high(self):
        """Test validation for weight > 1.0."""
        result = add_personality_directive("warmth", "Valid instruction", 1.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

    def test_add_personality_directive_validation_invalid_weight_string(self):
        """Test validation for string weight."""
        result = add_personality_directive("warmth", "Valid instruction", "invalid")
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

    def test_add_personality_directive_validation_boundary_weights(self):
        """Test validation accepts boundary weight values."""
        # These should pass validation (but will fail on DB since we're not mocking)
        with patch(
            "alpha_recall.tools.add_personality_directive.get_memgraph_service"
        ) as mock_get_service:
            mock_get_service.side_effect = Exception(
                "Expected - testing validation only"
            )

            # Test 0.0 weight
            result = add_personality_directive("warmth", "Valid instruction", 0.0)
            data = json.loads(result)
            assert "Weight must be a number" not in data["error"]

            # Test 1.0 weight
            result = add_personality_directive("warmth", "Valid instruction", 1.0)
            data = json.loads(result)
            assert "Weight must be a number" not in data["error"]

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_database_error(self, mock_get_service):
        """Test handling of database errors."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db
        mock_db.execute_and_fetch.side_effect = Exception("Database connection failed")
        mock_get_service.return_value = mock_service

        result = add_personality_directive("warmth", "Valid instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Database connection failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_service_initialization_error(
        self, mock_get_service
    ):
        """Test handling of service initialization errors."""
        mock_get_service.side_effect = Exception(
            "Failed to initialize Memgraph service"
        )

        result = add_personality_directive("warmth", "Valid instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Failed to initialize Memgraph service" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_creation_returns_no_results(
        self, mock_get_service
    ):
        """Test handling when directive creation returns no results."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait exists, no duplicates, but creation fails
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [],  # Duplicate check
            [],  # Creation returns empty (failure case)
        ]
        mock_get_service.return_value = mock_service

        result = add_personality_directive("warmth", "Some instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Failed to create directive" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_instruction_trimming(self, mock_get_service):
        """Test that instruction text is properly trimmed."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock successful flow
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [],  # Duplicate check
            [
                {  # Creation result
                    "directive_instruction": "Trimmed instruction",
                    "directive_weight": 0.5,
                    "directive_created_at": "2025-07-11T16:00:00+00:00",
                    "trait_name": "warmth",
                    "trait_description": "Caring patterns",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        # Call with padded whitespace
        result = add_personality_directive(
            "warmth", "  \n  Trimmed instruction  \t  ", 0.5
        )
        data = json.loads(result)

        assert data["success"] is True

        # Verify the trimmed instruction was used in duplicate check
        duplicate_check_call = mock_db.execute_and_fetch.call_args_list[1]
        params = duplicate_check_call[0][1]
        assert params["instruction"] == "Trimmed instruction"

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_datetime_serialization(self, mock_get_service):
        """Test proper datetime serialization in response."""
        from datetime import datetime

        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock with datetime object
        test_datetime = datetime(2025, 7, 11, 16, 30, 45, 123456)
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [],  # Duplicate check
            [
                {  # Creation result with datetime object
                    "directive_instruction": "Test instruction",
                    "directive_weight": 0.5,
                    "directive_created_at": test_datetime,
                    "trait_name": "warmth",
                    "trait_description": "Caring patterns",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = add_personality_directive("warmth", "Test instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is True
        # Should be converted to ISO format string with Z suffix
        assert (
            data["directive_added"]["created_at"] == "2025-07-11T16:30:45.123456+00:00"
        )

    @patch("alpha_recall.tools.add_personality_directive.get_memgraph_service")
    def test_add_personality_directive_parameter_binding_safety(self, mock_get_service):
        """Test that parameters are safely bound to prevent injection."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait exists check (should handle injection-like input safely)
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait not found
            [],  # Available traits (empty for simplicity)
        ]
        mock_get_service.return_value = mock_service

        # Use potentially problematic input
        malicious_trait = "warmth'; DROP TABLE nodes; --"
        malicious_instruction = "Be nice'; CREATE (evil:Hacker); --"

        result = add_personality_directive(malicious_trait, malicious_instruction, 0.5)
        data = json.loads(result)

        # Should handle safely (trait not found)
        assert data["success"] is False

        # Verify parameters were passed safely to database
        trait_check_call = mock_db.execute_and_fetch.call_args_list[0]
        params = trait_check_call[0][1]
        assert params["trait_name"] == malicious_trait  # Should be safely parameterized

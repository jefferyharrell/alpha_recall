"""Unit tests for update_personality_directive_weight tool."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.update_personality_directive_weight import (
    update_personality_directive_weight,
)


class TestUpdatePersonalityDirectiveWeight:
    """Test cases for update_personality_directive_weight tool."""

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_success(self, mock_get_service):
        """Test successful directive weight update."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait exists, directive exists, update succeeds
        mock_db.execute_and_fetch.side_effect = [
            [
                {
                    "trait_name": "warmth",
                    "trait_description": "Caring behavioral patterns",
                }
            ],  # Trait check
            [
                {
                    "instruction": "Be empathetic in conversations",
                    "current_weight": 0.6,
                    "created_at": "2025-07-11T16:00:00+00:00",
                }
            ],  # Directive check
            [
                {  # Update result
                    "directive_instruction": "Be empathetic in conversations",
                    "directive_weight": 0.9,
                    "directive_created_at": "2025-07-11T16:00:00+00:00",
                    "directive_last_updated": "2025-07-11T16:30:00+00:00",
                    "trait_name": "warmth",
                    "trait_description": "Caring behavioral patterns",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight(
            "warmth", "Be empathetic in conversations", 0.9
        )
        data = json.loads(result)

        assert data["success"] is True
        assert data["trait_name"] == "warmth"
        assert data["trait_description"] == "Caring behavioral patterns"
        assert (
            data["directive_updated"]["instruction"] == "Be empathetic in conversations"
        )
        assert data["directive_updated"]["previous_weight"] == 0.6
        assert data["directive_updated"]["new_weight"] == 0.9
        assert abs(data["directive_updated"]["weight_change"] - 0.3) < 0.0001
        assert "created_at" in data["directive_updated"]
        assert "last_updated" in data["directive_updated"]

        # Verify all queries were called
        assert mock_db.execute_and_fetch.call_count == 3

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_no_change(self, mock_get_service):
        """Test when weight is already the target value."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait exists, directive exists with same weight
        mock_db.execute_and_fetch.side_effect = [
            [
                {
                    "trait_name": "warmth",
                    "trait_description": "Caring behavioral patterns",
                }
            ],  # Trait check
            [
                {
                    "instruction": "Be empathetic in conversations",
                    "current_weight": 0.8,  # Same as target
                    "created_at": "2025-07-11T16:00:00+00:00",
                }
            ],  # Directive check
        ]
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight(
            "warmth", "Be empathetic in conversations", 0.8
        )
        data = json.loads(result)

        assert data["success"] is True
        assert "Weight unchanged" in data["message"]
        assert data["directive"]["weight"] == 0.8
        assert data["weight_change"] == 0.0

        # Should not call update query
        assert mock_db.execute_and_fetch.call_count == 2

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_trait_not_found(
        self, mock_get_service
    ):
        """Test updating directive in nonexistent trait."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait doesn't exist
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check - empty result
            [  # Available traits
                {"name": "communication_style"},
                {"name": "intellectual_engagement"},
                {"name": "problem_solving"},
            ],
        ]
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight(
            "nonexistent", "Some instruction", 0.5
        )
        data = json.loads(result)

        assert data["success"] is False
        assert "not found" in data["error"]
        assert "available_traits" in data
        assert len(data["available_traits"]) == 3
        assert "problem_solving" in data["available_traits"]
        assert "correlation_id" in data

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_directive_not_found(
        self, mock_get_service
    ):
        """Test updating nonexistent directive."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait exists but directive doesn't
        mock_db.execute_and_fetch.side_effect = [
            [
                {
                    "trait_name": "warmth",
                    "trait_description": "Caring behavioral patterns",
                }
            ],  # Trait check
            [],  # Directive check - empty result
            [  # Available directives
                {"instruction": "Be kind", "weight": 0.8},
                {"instruction": "Show empathy", "weight": 0.7},
            ],
        ]
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight(
            "warmth", "Nonexistent instruction", 0.5
        )
        data = json.loads(result)

        assert data["success"] is False
        assert "not found" in data["error"]
        assert "Nonexistent instruction" in data["error"]
        assert "available_directives" in data
        assert len(data["available_directives"]) == 2
        assert data["available_directives"][0]["instruction"] == "Be kind"
        assert "correlation_id" in data

    def test_update_personality_directive_weight_validation_empty_trait_name(self):
        """Test validation for empty trait name."""
        result = update_personality_directive_weight("", "Some instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Trait name cannot be empty" in data["error"]
        assert "correlation_id" in data

    def test_update_personality_directive_weight_validation_whitespace_trait_name(
        self,
    ):
        """Test validation for whitespace-only trait name."""
        result = update_personality_directive_weight("   ", "Some instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Trait name cannot be empty" in data["error"]

    def test_update_personality_directive_weight_validation_empty_instruction(self):
        """Test validation for empty instruction."""
        result = update_personality_directive_weight("warmth", "", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Instruction cannot be empty" in data["error"]
        assert "correlation_id" in data

    def test_update_personality_directive_weight_validation_whitespace_instruction(
        self,
    ):
        """Test validation for whitespace-only instruction."""
        result = update_personality_directive_weight("warmth", "   \n\t   ", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Instruction cannot be empty" in data["error"]

    def test_update_personality_directive_weight_validation_invalid_weight_negative(
        self,
    ):
        """Test validation for negative weight."""
        result = update_personality_directive_weight(
            "warmth", "Valid instruction", -0.1
        )
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]
        assert "correlation_id" in data

    def test_update_personality_directive_weight_validation_invalid_weight_too_high(
        self,
    ):
        """Test validation for weight > 1.0."""
        result = update_personality_directive_weight("warmth", "Valid instruction", 1.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

    def test_update_personality_directive_weight_validation_invalid_weight_string(self):
        """Test validation for string weight."""
        result = update_personality_directive_weight(
            "warmth", "Valid instruction", "invalid"
        )
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

    def test_update_personality_directive_weight_validation_boundary_weights(self):
        """Test validation accepts boundary weight values."""
        # These should pass validation (but will fail on DB since we're not mocking)
        with patch(
            "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
        ) as mock_get_service:
            mock_get_service.side_effect = Exception(
                "Expected - testing validation only"
            )

            # Test 0.0 weight
            result = update_personality_directive_weight(
                "warmth", "Valid instruction", 0.0
            )
            data = json.loads(result)
            assert "Weight must be a number" not in data["error"]

            # Test 1.0 weight
            result = update_personality_directive_weight(
                "warmth", "Valid instruction", 1.0
            )
            data = json.loads(result)
            assert "Weight must be a number" not in data["error"]

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_database_error(self, mock_get_service):
        """Test handling of database errors."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db
        mock_db.execute_and_fetch.side_effect = Exception("Database connection failed")
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight("warmth", "Valid instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Database connection failed" in data["error"]
        assert "correlation_id" in data

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_service_initialization_error(
        self, mock_get_service
    ):
        """Test handling of service initialization errors."""
        mock_get_service.side_effect = Exception(
            "Failed to initialize Memgraph service"
        )

        result = update_personality_directive_weight("warmth", "Valid instruction", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Failed to initialize Memgraph service" in data["error"]
        assert "correlation_id" in data

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_update_returns_no_results(
        self, mock_get_service
    ):
        """Test handling when update operation returns no results."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait exists, directive exists, but update fails
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [
                {"instruction": "Be nice", "current_weight": 0.5, "created_at": "now"}
            ],  # Directive check
            [],  # Update returns empty (failure case)
        ]
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight("warmth", "Be nice", 0.8)
        data = json.loads(result)

        assert data["success"] is False
        assert "Failed to update directive weight" in data["error"]
        assert "correlation_id" in data

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_input_trimming(self, mock_get_service):
        """Test that trait name and instruction are properly trimmed."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock successful flow
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [
                {
                    "instruction": "Trimmed instruction",
                    "current_weight": 0.5,
                    "created_at": "now",
                }
            ],  # Directive check
            [
                {  # Update result
                    "directive_instruction": "Trimmed instruction",
                    "directive_weight": 0.8,
                    "directive_created_at": "2025-07-11T16:00:00+00:00",
                    "directive_last_updated": "2025-07-11T16:30:00+00:00",
                    "trait_name": "warmth",
                    "trait_description": "Caring patterns",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        # Call with padded whitespace
        result = update_personality_directive_weight(
            "  \n  warmth  \t  ", "  \n  Trimmed instruction  \t  ", 0.8
        )
        data = json.loads(result)

        assert data["success"] is True

        # Verify the trimmed values were used in trait check
        trait_check_call = mock_db.execute_and_fetch.call_args_list[0]
        params = trait_check_call[0][1]
        assert params["trait_name"] == "warmth"

        # Verify the trimmed values were used in directive check
        directive_check_call = mock_db.execute_and_fetch.call_args_list[1]
        params = directive_check_call[0][1]
        assert params["trait_name"] == "warmth"
        assert params["instruction"] == "Trimmed instruction"

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_datetime_serialization(
        self, mock_get_service
    ):
        """Test proper datetime serialization in response."""
        from datetime import datetime

        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock with datetime objects
        created_datetime = datetime(2025, 7, 11, 16, 0, 0, 123456)
        updated_datetime = datetime(2025, 7, 11, 16, 30, 45, 987654)
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [
                {"instruction": "Be nice", "current_weight": 0.5, "created_at": "now"}
            ],  # Directive check
            [
                {  # Update result with datetime objects
                    "directive_instruction": "Be nice",
                    "directive_weight": 0.8,
                    "directive_created_at": created_datetime,
                    "directive_last_updated": updated_datetime,
                    "trait_name": "warmth",
                    "trait_description": "Caring patterns",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight("warmth", "Be nice", 0.8)
        data = json.loads(result)

        assert data["success"] is True
        # Should be converted to ISO format strings with Z suffix
        assert (
            data["directive_updated"]["created_at"]
            == "2025-07-11T16:00:00.123456+00:00"
        )
        assert (
            data["directive_updated"]["last_updated"]
            == "2025-07-11T16:30:45.987654+00:00"
        )

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_parameter_binding_safety(
        self, mock_get_service
    ):
        """Test that parameters are safely bound to prevent injection."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait check (should handle injection-like input safely)
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait not found
            [],  # Available traits (empty for simplicity)
        ]
        mock_get_service.return_value = mock_service

        # Use potentially problematic input
        malicious_trait = "warmth'; DROP TABLE nodes; --"
        malicious_instruction = "Be nice'; CREATE (evil:Hacker); --"

        result = update_personality_directive_weight(
            malicious_trait, malicious_instruction, 0.5
        )
        data = json.loads(result)

        # Should handle safely (trait not found)
        assert data["success"] is False

        # Verify parameters were passed safely to database
        trait_check_call = mock_db.execute_and_fetch.call_args_list[0]
        params = trait_check_call[0][1]
        assert params["trait_name"] == malicious_trait  # Should be safely parameterized

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_float_conversion(
        self, mock_get_service
    ):
        """Test that weight is properly converted to float for database storage."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock successful flow
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [
                {"instruction": "Be nice", "current_weight": 0.5, "created_at": "now"}
            ],  # Directive check
            [
                {
                    "directive_instruction": "Be nice",
                    "directive_weight": 1.0,
                    "directive_created_at": "2025-07-11T16:00:00+00:00",
                    "directive_last_updated": "2025-07-11T16:30:00+00:00",
                    "trait_name": "warmth",
                    "trait_description": "Caring patterns",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        # Call with integer weight
        result = update_personality_directive_weight("warmth", "Be nice", 1)
        data = json.loads(result)

        assert data["success"] is True

        # Verify weight was converted to float for database
        update_call = mock_db.execute_and_fetch.call_args_list[2]
        params = update_call[0][1]
        assert isinstance(params["new_weight"], float)
        assert params["new_weight"] == 1.0

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_datetime_fallback(
        self, mock_get_service
    ):
        """Test datetime serialization fallback for non-datetime objects."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock with string timestamps (fallback case)
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [
                {"instruction": "Be nice", "current_weight": 0.5, "created_at": "now"}
            ],  # Directive check
            [
                {
                    "directive_instruction": "Be nice",
                    "directive_weight": 0.8,
                    "directive_created_at": "2025-07-11T16:00:00",  # String, not datetime
                    "directive_last_updated": "2025-07-11T16:30:00",  # String, not datetime
                    "trait_name": "warmth",
                    "trait_description": "Caring patterns",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight("warmth", "Be nice", 0.8)
        data = json.loads(result)

        assert data["success"] is True
        # Should handle string timestamps gracefully and convert to proper UTC format
        assert data["directive_updated"]["created_at"] == "2025-07-11T16:00:00+00:00"
        assert data["directive_updated"]["last_updated"] == "2025-07-11T16:30:00+00:00"

    @patch(
        "alpha_recall.tools.update_personality_directive_weight.get_memgraph_service"
    )
    def test_update_personality_directive_weight_precision_handling(
        self, mock_get_service
    ):
        """Test weight change calculation with floating point precision."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock with very close weight values (precision test)
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "warmth", "trait_description": "Caring patterns"}
            ],  # Trait check
            [
                {
                    "instruction": "Be nice",
                    "current_weight": 0.80000001,  # Very close to target
                    "created_at": "now",
                }
            ],  # Directive check
        ]
        mock_get_service.return_value = mock_service

        result = update_personality_directive_weight("warmth", "Be nice", 0.8)
        data = json.loads(result)

        assert data["success"] is True
        assert "Weight unchanged" in data["message"]
        # Should detect that weights are effectively the same (within 0.0001 tolerance)
        assert data["weight_change"] == 0.0

        # Should not call update query due to tolerance check
        assert mock_db.execute_and_fetch.call_count == 2

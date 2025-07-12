"""Unit tests for create_personality_trait tool."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.create_personality_trait import create_personality_trait


class TestCreatePersonalityTrait:
    """Test cases for create_personality_trait tool."""

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_success_with_existing_root(
        self, mock_get_service
    ):
        """Test successful trait creation when Agent_Personality root already exists."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait doesn't exist, root exists, creation succeeds
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check - doesn't exist
            [{"root": "mock_root"}],  # Root exists
            [
                {  # Successful creation
                    "trait_name": "curiosity",
                    "trait_description": "Drive to explore and question",
                    "trait_weight": 0.9,
                    "trait_created_at": "2025-07-11T16:00:00+00:00",
                    "root_name": "Alpha Core Identity",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = create_personality_trait(
            "curiosity", "Drive to explore and question", 0.9
        )
        data = json.loads(result)

        assert data["success"] is True
        assert data["trait_created"]["name"] == "curiosity"
        assert data["trait_created"]["description"] == "Drive to explore and question"
        assert data["trait_created"]["weight"] == 0.9
        assert "created_at" in data["trait_created"]
        assert data["linked_to_root"] == "Alpha Core Identity"

        # Verify queries: trait check + root check + creation
        assert mock_db.execute_and_fetch.call_count == 3

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_success_with_root_creation(
        self, mock_get_service
    ):
        """Test successful trait creation when Agent_Personality root needs to be created."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait doesn't exist, root doesn't exist, root creation + trait creation succeed
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check - doesn't exist
            [],  # Root doesn't exist
            [{"root_name": "Alpha Core Identity"}],  # Root creation successful
            [
                {  # Trait creation successful
                    "trait_name": "empathy",
                    "trait_description": "Understanding others' emotions",
                    "trait_weight": 0.8,
                    "trait_created_at": "2025-07-11T16:00:00+00:00",
                    "root_name": "Alpha Core Identity",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = create_personality_trait(
            "empathy", "Understanding others' emotions", 0.8
        )
        data = json.loads(result)

        assert data["success"] is True
        assert data["trait_created"]["name"] == "empathy"
        assert data["trait_created"]["description"] == "Understanding others' emotions"
        assert data["trait_created"]["weight"] == 0.8
        assert data["linked_to_root"] == "Alpha Core Identity"

        # Verify queries: trait check + root check + root creation + trait creation
        assert mock_db.execute_and_fetch.call_count == 4

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_default_weight(self, mock_get_service):
        """Test trait creation with default weight (1.0)."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock successful flow with default weight
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check
            [{"root": "exists"}],  # Root exists
            [
                {
                    "trait_name": "default_weight_trait",
                    "trait_description": "Test description",
                    "trait_weight": 1.0,  # Default weight
                    "trait_created_at": "2025-07-11T16:00:00+00:00",
                    "root_name": "Alpha Core Identity",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        # Call without weight parameter (should default to 1.0)
        result = create_personality_trait("default_weight_trait", "Test description")
        data = json.loads(result)

        assert data["success"] is True
        assert data["trait_created"]["weight"] == 1.0

        # Verify the weight parameter was passed as 1.0 to the database
        creation_call = mock_db.execute_and_fetch.call_args_list[2]
        params = creation_call[0][1]
        assert params["weight"] == 1.0

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_duplicate_prevention(self, mock_get_service):
        """Test prevention of duplicate trait creation."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait already exists
        mock_db.execute_and_fetch.side_effect = [
            [{"trait_name": "existing_trait"}],  # Trait exists
        ]
        mock_get_service.return_value = mock_service

        result = create_personality_trait("existing_trait", "Some description", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "already exists" in data["error"]
        assert "existing_trait" in data
        assert data["existing_trait"]["name"] == "existing_trait"
        assert "correlation_id" in data

        # Should only call trait check
        assert mock_db.execute_and_fetch.call_count == 1

    def test_create_personality_trait_validation_empty_trait_name(self):
        """Test validation for empty trait name."""
        result = create_personality_trait("", "Valid description", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Trait name cannot be empty" in data["error"]
        assert "correlation_id" in data

    def test_create_personality_trait_validation_whitespace_trait_name(self):
        """Test validation for whitespace-only trait name."""
        result = create_personality_trait("   \n\t   ", "Valid description", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Trait name cannot be empty" in data["error"]

    def test_create_personality_trait_validation_empty_description(self):
        """Test validation for empty description."""
        result = create_personality_trait("valid_name", "", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Description cannot be empty" in data["error"]
        assert "correlation_id" in data

    def test_create_personality_trait_validation_whitespace_description(self):
        """Test validation for whitespace-only description."""
        result = create_personality_trait("valid_name", "   \n\t   ", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Description cannot be empty" in data["error"]

    def test_create_personality_trait_validation_invalid_weight_negative(self):
        """Test validation for negative weight."""
        result = create_personality_trait("valid_name", "Valid description", -0.1)
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]
        assert "correlation_id" in data

    def test_create_personality_trait_validation_invalid_weight_too_high(self):
        """Test validation for weight > 1.0."""
        result = create_personality_trait("valid_name", "Valid description", 1.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

    def test_create_personality_trait_validation_invalid_weight_string(self):
        """Test validation for string weight."""
        result = create_personality_trait("valid_name", "Valid description", "invalid")
        data = json.loads(result)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

    def test_create_personality_trait_validation_boundary_weights(self):
        """Test validation accepts boundary weight values."""
        # These should pass validation (but will fail on DB since we're not mocking)
        with patch(
            "alpha_recall.tools.create_personality_trait.get_memgraph_service"
        ) as mock_get_service:
            mock_get_service.side_effect = Exception(
                "Expected - testing validation only"
            )

            # Test 0.0 weight
            result = create_personality_trait("valid_name", "Valid description", 0.0)
            data = json.loads(result)
            assert "Weight must be a number" not in data["error"]

            # Test 1.0 weight
            result = create_personality_trait("valid_name", "Valid description", 1.0)
            data = json.loads(result)
            assert "Weight must be a number" not in data["error"]

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_database_error(self, mock_get_service):
        """Test handling of database errors."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db
        mock_db.execute_and_fetch.side_effect = Exception("Database connection failed")
        mock_get_service.return_value = mock_service

        result = create_personality_trait("valid_name", "Valid description", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Database connection failed" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_service_initialization_error(
        self, mock_get_service
    ):
        """Test handling of service initialization errors."""
        mock_get_service.side_effect = Exception(
            "Failed to initialize Memgraph service"
        )

        result = create_personality_trait("valid_name", "Valid description", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Failed to initialize Memgraph service" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_creation_returns_no_results(
        self, mock_get_service
    ):
        """Test handling when trait creation returns no results."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait doesn't exist, root exists, but creation fails
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check
            [{"root": "exists"}],  # Root exists
            [],  # Creation returns empty (failure case)
        ]
        mock_get_service.return_value = mock_service

        result = create_personality_trait("test_trait", "Test description", 0.5)
        data = json.loads(result)

        assert data["success"] is False
        assert "Failed to create trait" in data["error"]
        assert "correlation_id" in data

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_input_trimming(self, mock_get_service):
        """Test that trait name and description are properly trimmed."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock successful flow
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check
            [{"root": "exists"}],  # Root exists
            [
                {  # Creation result
                    "trait_name": "trimmed_trait",
                    "trait_description": "Trimmed description",
                    "trait_weight": 0.5,
                    "trait_created_at": "2025-07-11T16:00:00+00:00",
                    "root_name": "Alpha Core Identity",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        # Call with padded whitespace
        result = create_personality_trait(
            "  \n  trimmed_trait  \t  ", "  \n  Trimmed description  \t  ", 0.5
        )
        data = json.loads(result)

        assert data["success"] is True

        # Verify the trimmed values were used in trait check
        trait_check_call = mock_db.execute_and_fetch.call_args_list[0]
        params = trait_check_call[0][1]
        assert params["trait_name"] == "trimmed_trait"

        # Verify the trimmed values were used in creation
        creation_call = mock_db.execute_and_fetch.call_args_list[2]
        params = creation_call[0][1]
        assert params["trait_name"] == "trimmed_trait"
        assert params["description"] == "Trimmed description"

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_datetime_serialization(self, mock_get_service):
        """Test proper datetime serialization in response."""
        from datetime import datetime

        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock with datetime object
        test_datetime = datetime(2025, 7, 11, 16, 30, 45, 123456)
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check
            [{"root": "exists"}],  # Root exists
            [
                {  # Creation result with datetime object
                    "trait_name": "test_trait",
                    "trait_description": "Test description",
                    "trait_weight": 0.5,
                    "trait_created_at": test_datetime,
                    "root_name": "Alpha Core Identity",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = create_personality_trait("test_trait", "Test description", 0.5)
        data = json.loads(result)

        assert data["success"] is True
        # Should be converted to ISO format string with Z suffix
        assert data["trait_created"]["created_at"] == "2025-07-11T16:30:45.123456+00:00"

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_parameter_binding_safety(self, mock_get_service):
        """Test that parameters are safely bound to prevent injection."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock trait check (should handle injection-like input safely)
        mock_db.execute_and_fetch.side_effect = [
            [
                {"trait_name": "malicious_trait"}
            ],  # Trait "exists" (safe parameterization)
        ]
        mock_get_service.return_value = mock_service

        # Use potentially problematic input
        malicious_trait = "trait'; DROP TABLE nodes; --"
        malicious_description = "description'; CREATE (evil:Hacker); --"

        result = create_personality_trait(malicious_trait, malicious_description, 0.5)
        data = json.loads(result)

        # Should handle safely (trait "already exists")
        assert data["success"] is False

        # Verify parameters were passed safely to database
        trait_check_call = mock_db.execute_and_fetch.call_args_list[0]
        params = trait_check_call[0][1]
        assert params["trait_name"] == malicious_trait  # Should be safely parameterized

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_float_weight_conversion(self, mock_get_service):
        """Test that weight is properly converted to float for database storage."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock successful flow
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check
            [{"root": "exists"}],  # Root exists
            [
                {
                    "trait_name": "test_trait",
                    "trait_description": "Test description",
                    "trait_weight": 0.75,
                    "trait_created_at": "2025-07-11T16:00:00+00:00",
                    "root_name": "Alpha Core Identity",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        # Call with integer weight
        result = create_personality_trait("test_trait", "Test description", 1)
        data = json.loads(result)

        assert data["success"] is True

        # Verify weight was converted to float for database
        creation_call = mock_db.execute_and_fetch.call_args_list[2]
        params = creation_call[0][1]
        assert isinstance(params["weight"], float)
        assert params["weight"] == 1.0

    @patch("alpha_recall.tools.create_personality_trait.get_memgraph_service")
    def test_create_personality_trait_datetime_fallback(self, mock_get_service):
        """Test datetime serialization fallback for non-datetime objects."""
        mock_service = Mock()
        mock_db = Mock()
        mock_service.db = mock_db

        # Mock with string timestamp (fallback case)
        mock_db.execute_and_fetch.side_effect = [
            [],  # Trait check
            [{"root": "exists"}],  # Root exists
            [
                {
                    "trait_name": "test_trait",
                    "trait_description": "Test description",
                    "trait_weight": 0.5,
                    "trait_created_at": "2025-07-11T16:00:00",  # String, not datetime
                    "root_name": "Alpha Core Identity",
                }
            ],
        ]
        mock_get_service.return_value = mock_service

        result = create_personality_trait("test_trait", "Test description", 0.5)
        data = json.loads(result)

        assert data["success"] is True
        # Should handle string timestamp gracefully and convert to proper UTC format
        assert data["trait_created"]["created_at"] == "2025-07-11T16:00:00+00:00"

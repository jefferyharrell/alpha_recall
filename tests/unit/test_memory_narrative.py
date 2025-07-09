"""Unit tests for narrative memory tools."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.tools.browse_narrative import browse_narrative
from alpha_recall.tools.recall_narrative import recall_narrative
from alpha_recall.tools.remember_narrative import remember_narrative
from alpha_recall.tools.search_narratives import search_narratives


class TestRememberNarrative(unittest.TestCase):
    """Test cases for remember_narrative tool."""

    @patch("alpha_recall.tools.remember_narrative.get_narrative_service")
    def test_remember_narrative_basic_functionality(self, mock_get_service):
        """Test basic narrative storage with all required parameters."""
        # Mock the service response
        mock_service = AsyncMock()
        mock_service.store_story.return_value = {
            "success": True,
            "story_id": "story_1234567890_abc123",
            "title": "The Great Debugging Session",
            "created_at": "2025-07-09T08:00:00+00:00",
            "paragraph_count": 3,
            "embeddings_generated": 8,
            "storage_location": "hybrid_redis_memgraph",
            "correlation_id": "narrative_store_12345678",
        }
        mock_get_service.return_value = mock_service

        title = "The Great Debugging Session"
        paragraphs = [
            "We started with a mysterious bug that was causing random crashes.",
            "After hours of investigation, we discovered it was a race condition.",
            "The fix was surprisingly simple - just a single line change.",
        ]
        participants = ["Alpha", "Jeffery"]

        result = remember_narrative(title, paragraphs, participants)
        response_data = json.loads(result)

        # Verify response structure
        self.assertIn("success", response_data)
        self.assertTrue(response_data["success"])
        self.assertIn("story", response_data)
        self.assertIn("processing", response_data)
        self.assertIn("correlation_id", response_data)

        # Verify story data
        story = response_data["story"]
        self.assertEqual(story["title"], title)
        self.assertEqual(story["participants"], participants)
        self.assertEqual(story["paragraph_count"], len(paragraphs))
        self.assertEqual(story["outcome"], "ongoing")  # default
        self.assertEqual(story["tags"], [])  # default
        self.assertEqual(story["references"], [])  # default
        self.assertIn("story_id", story)
        self.assertIn("created_at", story)

        # Verify processing data
        processing = response_data["processing"]
        self.assertEqual(processing["paragraphs_processed"], len(paragraphs))
        self.assertEqual(processing["storage_location"], "hybrid_redis_memgraph")

        # Verify story_id format
        story_id = story["story_id"]
        self.assertTrue(story_id.startswith("story_"))
        self.assertIn("_", story_id[6:])  # Should have timestamp and correlation parts

    @patch("alpha_recall.tools.remember_narrative.get_narrative_service")
    def test_remember_narrative_with_optional_parameters(self, mock_get_service):
        """Test narrative storage with all optional parameters."""
        # Mock the service response
        mock_service = AsyncMock()
        mock_service.store_story.return_value = {
            "success": True,
            "story_id": "story_1234567890_abc123",
            "title": "Project Alpha Breakthrough",
            "created_at": "2025-07-09T08:00:00+00:00",
            "paragraph_count": 2,
            "embeddings_generated": 4,
            "storage_location": "hybrid_redis_memgraph",
            "correlation_id": "narrative_abc12345",
        }
        mock_get_service.return_value = mock_service

        title = "Project Alpha Breakthrough"
        paragraphs = [
            "We had been struggling with memory architecture for weeks.",
            "Then Alpha suggested a three-tier approach that changed everything.",
        ]
        participants = ["Alpha", "Jeffery", "Kylee"]
        outcome = "breakthrough"
        tags = ["memory", "architecture", "collaboration"]
        references = ["story_123456_abc", "story_789012_def"]

        result = remember_narrative(
            title=title,
            paragraphs=paragraphs,
            participants=participants,
            outcome=outcome,
            tags=tags,
            references=references,
        )
        response_data = json.loads(result)

        # Verify all optional parameters are preserved
        story = response_data["story"]
        self.assertEqual(story["outcome"], outcome)
        self.assertEqual(story["tags"], tags)
        self.assertEqual(story["references"], references)
        self.assertEqual(story["participants"], participants)

    @patch("alpha_recall.tools.remember_narrative.get_narrative_service")
    def test_remember_narrative_data_cleaning(self, mock_get_service):
        """Test that input data is properly cleaned."""
        # Mock the service response
        mock_service = AsyncMock()
        mock_service.store_story.return_value = {
            "success": True,
            "story_id": "story_1234567890_abc123",
            "title": "  Clean Data Test  ",
            "created_at": "2025-07-09T08:00:00+00:00",
            "paragraph_count": 2,
            "embeddings_generated": 4,
            "storage_location": "hybrid_redis_memgraph",
            "correlation_id": "narrative_abc12345",
        }
        mock_get_service.return_value = mock_service

        title = "  Clean Data Test  "
        paragraphs = [
            "  First paragraph  ",
            "",  # Empty paragraph should be filtered out
            "   ",  # Whitespace-only paragraph should be filtered out
            "Second paragraph",
        ]
        participants = [
            "  Alpha  ",
            "",  # Empty participant should be filtered out
            "Jeffery",
        ]
        tags = [
            "  tag1  ",
            "",  # Empty tag should be filtered out
            "tag2",
        ]
        references = [
            "  ref1  ",
            "",  # Empty reference should be filtered out
            "ref2",
        ]

        result = remember_narrative(
            title=title,
            paragraphs=paragraphs,
            participants=participants,
            tags=tags,
            references=references,
        )
        response_data = json.loads(result)

        # Verify data cleaning
        story = response_data["story"]
        processing = response_data["processing"]

        # Title should be cleaned but preserve the original for consistency
        self.assertEqual(story["title"], title)  # Original title preserved

        # Only non-empty paragraphs should be counted
        self.assertEqual(processing["paragraphs_processed"], 2)

        # Cleaned participants
        self.assertEqual(story["participants"], ["Alpha", "Jeffery"])

        # Cleaned tags
        self.assertEqual(story["tags"], ["tag1", "tag2"])

        # Cleaned references
        self.assertEqual(story["references"], ["ref1", "ref2"])

    def test_remember_narrative_validation_empty_title(self):
        """Test validation for empty title."""
        result = remember_narrative(
            title="",
            paragraphs=["Some content"],
            participants=["Alpha"],
        )
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("error", response_data)
        self.assertIn("Title cannot be empty", response_data["error"])

    def test_remember_narrative_validation_empty_paragraphs(self):
        """Test validation for empty paragraphs."""
        result = remember_narrative(
            title="Test Story",
            paragraphs=[],
            participants=["Alpha"],
        )
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("error", response_data)
        self.assertIn(
            "At least one non-empty paragraph required", response_data["error"]
        )

    def test_remember_narrative_validation_whitespace_only_paragraphs(self):
        """Test validation for paragraphs with only whitespace."""
        result = remember_narrative(
            title="Test Story",
            paragraphs=["   ", "\t\n", ""],
            participants=["Alpha"],
        )
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("error", response_data)
        self.assertIn(
            "At least one non-empty paragraph required", response_data["error"]
        )

    def test_remember_narrative_validation_empty_participants(self):
        """Test validation for empty participants."""
        result = remember_narrative(
            title="Test Story",
            paragraphs=["Some content"],
            participants=[],
        )
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("error", response_data)
        self.assertIn("At least one participant required", response_data["error"])

    def test_remember_narrative_validation_whitespace_only_participants(self):
        """Test validation for participants with only whitespace."""
        result = remember_narrative(
            title="Test Story",
            paragraphs=["Some content"],
            participants=["   ", "\t\n", ""],
        )
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("error", response_data)
        self.assertIn("At least one participant required", response_data["error"])

    def test_remember_narrative_json_response_format(self):
        """Test that response is valid JSON."""
        result = remember_narrative(
            title="JSON Test",
            paragraphs=["Test content"],
            participants=["Alpha"],
        )

        # Should not raise an exception
        response_data = json.loads(result)

        # Should be properly formatted JSON
        self.assertIsInstance(response_data, dict)

        # Re-serialize to ensure it's valid JSON
        json_string = json.dumps(response_data)
        self.assertIsInstance(json_string, str)

    @patch("alpha_recall.tools.remember_narrative.generate_correlation_id")
    @patch("alpha_recall.tools.remember_narrative.get_narrative_service")
    def test_remember_narrative_correlation_id_format(
        self, mock_get_service, mock_generate_id
    ):
        """Test that correlation ID is properly formatted."""
        # Mock correlation ID generation
        mock_generate_id.return_value = "narrative_abc12345"

        # Mock the service response
        mock_service = AsyncMock()
        mock_service.store_story.return_value = {
            "success": True,
            "story_id": "story_1234567890_abc12345",
            "title": "Correlation ID Test",
            "created_at": "2025-07-09T08:00:00+00:00",
            "paragraph_count": 1,
            "embeddings_generated": 2,
            "storage_location": "hybrid_redis_memgraph",
            "correlation_id": "narrative_abc12345",
        }
        mock_get_service.return_value = mock_service

        result = remember_narrative(
            title="Correlation ID Test",
            paragraphs=["Test content"],
            participants=["Alpha"],
        )
        response_data = json.loads(result)

        correlation_id = response_data["correlation_id"]
        self.assertTrue(correlation_id.startswith("narrative_"))
        self.assertEqual(len(correlation_id), 18)  # "narrative_" + 8 chars

        # Story ID should include correlation ID suffix
        story_id = response_data["story"]["story_id"]
        correlation_suffix = correlation_id.split("_")[1]
        self.assertIn(correlation_suffix, story_id)

    @patch("alpha_recall.tools.remember_narrative.generate_correlation_id")
    @patch("alpha_recall.tools.remember_narrative.get_narrative_service")
    def test_remember_narrative_timestamp_consistency(
        self, mock_get_service, mock_generate_id
    ):
        """Test timestamp consistency between story_id and created_at."""
        # Mock correlation ID generation
        mock_generate_id.return_value = "narrative_abc12345"

        # Mock the service response with consistent timestamp
        mock_service = AsyncMock()
        mock_service.store_story.return_value = {
            "success": True,
            "story_id": "story_1752070566_abc12345",
            "title": "Timestamp Test",
            "created_at": "2025-07-09T08:00:00+00:00",
            "paragraph_count": 1,
            "embeddings_generated": 2,
            "storage_location": "hybrid_redis_memgraph",
            "correlation_id": "narrative_abc12345",
        }
        mock_get_service.return_value = mock_service

        result = remember_narrative(
            title="Timestamp Test",
            paragraphs=["Test content"],
            participants=["Alpha"],
        )
        response_data = json.loads(result)

        # Story ID should include the timestamp
        story_id = response_data["story"]["story_id"]
        self.assertIn("1752070566", story_id)

        # created_at should be in ISO format
        created_at = response_data["story"]["created_at"]
        self.assertTrue(created_at.endswith("+00:00"))  # UTC timezone
        self.assertIn("T", created_at)  # ISO format separator


class TestSearchNarratives(unittest.TestCase):
    """Test cases for search_narratives tool."""

    def test_search_narratives_basic_functionality(self):
        """Test basic narrative search with default parameters."""
        result = search_narratives("test query")
        response_data = json.loads(result)

        # Verify response structure
        self.assertIn("success", response_data)
        self.assertTrue(response_data["success"])
        self.assertIn("search", response_data)
        self.assertIn("results", response_data)
        self.assertIn("metadata", response_data)
        self.assertIn("correlation_id", response_data)

        # Verify search parameters
        search = response_data["search"]
        self.assertEqual(search["query"], "test query")
        self.assertEqual(search["search_type"], "semantic")  # default
        self.assertEqual(search["granularity"], "story")  # default
        self.assertEqual(search["limit"], 10)  # default

        # Verify metadata
        metadata = response_data["metadata"]
        self.assertEqual(metadata["results_count"], 0)
        self.assertEqual(metadata["search_method"], "vector_similarity")
        self.assertEqual(metadata["embedding_model"], "dual_semantic_emotional")

    def test_search_narratives_with_all_parameters(self):
        """Test search with all optional parameters."""
        result = search_narratives(
            query="complex query",
            search_type="both",
            granularity="paragraph",
            limit=20,
        )
        response_data = json.loads(result)

        search = response_data["search"]
        self.assertEqual(search["query"], "complex query")
        self.assertEqual(search["search_type"], "both")
        self.assertEqual(search["granularity"], "paragraph")
        self.assertEqual(search["limit"], 20)

    def test_search_narratives_validation_empty_query(self):
        """Test validation for empty query."""
        result = search_narratives("")
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("Query cannot be empty", response_data["error"])

    def test_search_narratives_validation_invalid_search_type(self):
        """Test validation for invalid search_type."""
        result = search_narratives("query", search_type="invalid")
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("search_type must be", response_data["error"])

    def test_search_narratives_validation_invalid_granularity(self):
        """Test validation for invalid granularity."""
        result = search_narratives("query", granularity="invalid")
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("granularity must be", response_data["error"])

    def test_search_narratives_validation_invalid_limit(self):
        """Test validation for invalid limit values."""
        # Test limit too small
        result = search_narratives("query", limit=0)
        response_data = json.loads(result)
        self.assertFalse(response_data["success"])
        self.assertIn("limit must be between", response_data["error"])

        # Test limit too large
        result = search_narratives("query", limit=101)
        response_data = json.loads(result)
        self.assertFalse(response_data["success"])
        self.assertIn("limit must be between", response_data["error"])


class TestRecallNarrative(unittest.TestCase):
    """Test cases for recall_narrative tool."""

    @patch("alpha_recall.tools.recall_narrative.get_narrative_service")
    def test_recall_narrative_basic_functionality(self, mock_get_service):
        """Test basic narrative recall with valid story_id."""
        # Mock the service response
        mock_service = AsyncMock()
        mock_service.get_story.return_value = {
            "story_id": "story_1234567890_abc123",
            "title": "Test Story",
            "created_at": "2025-07-09T08:00:00+00:00",
            "participants": ["Alpha", "Jeffery"],
            "tags": ["test", "example"],
            "outcome": "ongoing",
            "references": [],
            "paragraphs": [
                {"text": "First paragraph", "order": 0},
                {"text": "Second paragraph", "order": 1},
            ],
            "metadata": {
                "paragraph_count": 2,
                "embeddings_generated": 4,
                "storage_location": "hybrid_redis_memgraph",
            },
        }
        mock_get_service.return_value = mock_service

        story_id = "story_1234567890_abc123"
        result = recall_narrative(story_id)
        response_data = json.loads(result)

        # Verify response structure
        self.assertIn("success", response_data)
        self.assertTrue(response_data["success"])
        self.assertIn("story", response_data)
        self.assertIn("correlation_id", response_data)

        # Verify story structure
        story = response_data["story"]
        self.assertEqual(story["story_id"], story_id)
        self.assertIn("title", story)
        self.assertIn("created_at", story)
        self.assertIn("participants", story)
        self.assertIn("tags", story)
        self.assertIn("outcome", story)
        self.assertIn("references", story)
        self.assertIn("paragraphs", story)
        self.assertIn("metadata", story)

        # Verify paragraphs structure
        paragraphs = story["paragraphs"]
        self.assertIsInstance(paragraphs, list)
        for para in paragraphs:
            self.assertIn("order", para)
            self.assertIn("text", para)

    def test_recall_narrative_validation_empty_story_id(self):
        """Test validation for empty story_id."""
        result = recall_narrative("")
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("story_id cannot be empty", response_data["error"])

    def test_recall_narrative_validation_invalid_story_id_format(self):
        """Test validation for invalid story_id format."""
        result = recall_narrative("invalid_id")
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("story_id must start with 'story_'", response_data["error"])

    def test_recall_narrative_json_response_format(self):
        """Test that response is valid JSON."""
        result = recall_narrative("story_1234567890_abc123")

        # Should not raise an exception
        response_data = json.loads(result)
        self.assertIsInstance(response_data, dict)


class TestBrowseNarrative(unittest.TestCase):
    """Test cases for browse_narrative tool."""

    @patch("alpha_recall.tools.browse_narrative.get_narrative_service")
    def test_browse_narrative_basic_functionality(self, mock_get_service):
        """Test basic narrative browsing with default parameters."""
        # Mock the service response
        mock_service = AsyncMock()
        mock_service.list_stories.return_value = {
            "stories": [],
            "limit": 10,
            "offset": 0,
            "total_count": 0,
            "has_more": False,
            "returned_count": 0,
        }
        mock_get_service.return_value = mock_service

        result = browse_narrative()
        response_data = json.loads(result)

        # Verify response structure
        self.assertIn("success", response_data)
        self.assertTrue(response_data["success"])
        self.assertIn("browse_data", response_data)
        self.assertIn("correlation_id", response_data)

        # Verify browse_data structure
        browse_data = response_data["browse_data"]
        self.assertIn("stories", browse_data)
        self.assertIn("pagination", browse_data)
        self.assertIn("filters", browse_data)
        self.assertIn("metadata", browse_data)

        # Verify pagination
        pagination = browse_data["pagination"]
        self.assertEqual(pagination["limit"], 10)  # default
        self.assertEqual(pagination["offset"], 0)  # default
        self.assertEqual(pagination["total_count"], 0)  # placeholder
        self.assertFalse(pagination["has_more"])  # placeholder

        # Verify metadata
        metadata = browse_data["metadata"]
        self.assertEqual(metadata["sort_order"], "chronological_desc")
        self.assertEqual(metadata["storage_location"], "hybrid_redis_memgraph")

    def test_browse_narrative_with_all_parameters(self):
        """Test browsing with all optional parameters."""
        result = browse_narrative(
            limit=20,
            offset=10,
            since="7d",
            participants=["Alpha", "Jeffery"],
            tags=["breakthrough", "memory"],
            outcome="resolution",
        )
        response_data = json.loads(result)

        browse_data = response_data["browse_data"]

        # Verify pagination
        pagination = browse_data["pagination"]
        self.assertEqual(pagination["limit"], 20)
        self.assertEqual(pagination["offset"], 10)

        # Verify filters
        filters = browse_data["filters"]
        self.assertEqual(filters["since"], "7d")
        self.assertEqual(filters["participants"], ["Alpha", "Jeffery"])
        self.assertEqual(filters["tags"], ["breakthrough", "memory"])
        self.assertEqual(filters["outcome"], "resolution")

    def test_browse_narrative_parameter_cleaning(self):
        """Test that filter parameters are properly cleaned."""
        result = browse_narrative(
            participants=["  Alpha  ", "", "Jeffery"],
            tags=["  tag1  ", "", "tag2"],
            outcome="  resolution  ",
        )
        response_data = json.loads(result)

        filters = response_data["browse_data"]["filters"]

        # Verify cleaned participants
        self.assertEqual(filters["participants"], ["Alpha", "Jeffery"])

        # Verify cleaned tags
        self.assertEqual(filters["tags"], ["tag1", "tag2"])

        # Verify cleaned outcome
        self.assertEqual(filters["outcome"], "resolution")

    def test_browse_narrative_validation_invalid_limit(self):
        """Test validation for invalid limit values."""
        # Test limit too small
        result = browse_narrative(limit=0)
        response_data = json.loads(result)
        self.assertFalse(response_data["success"])
        self.assertIn("limit must be between", response_data["error"])

        # Test limit too large
        result = browse_narrative(limit=101)
        response_data = json.loads(result)
        self.assertFalse(response_data["success"])
        self.assertIn("limit must be between", response_data["error"])

    def test_browse_narrative_validation_invalid_offset(self):
        """Test validation for invalid offset values."""
        result = browse_narrative(offset=-1)
        response_data = json.loads(result)

        self.assertFalse(response_data["success"])
        self.assertIn("offset must be non-negative", response_data["error"])

    def test_browse_narrative_json_response_format(self):
        """Test that response is valid JSON."""
        result = browse_narrative()

        # Should not raise an exception
        response_data = json.loads(result)
        self.assertIsInstance(response_data, dict)


if __name__ == "__main__":
    unittest.main()

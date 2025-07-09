"""Integration tests for narrative memory tools."""

import json
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastmcp import FastMCP

from alpha_recall.tools.browse_narrative import browse_narrative
from alpha_recall.tools.recall_narrative import recall_narrative
from alpha_recall.tools.remember_narrative import remember_narrative
from alpha_recall.tools.search_narratives import search_narratives


class TestNarrativeMemoryIntegration:
    """Integration tests for narrative memory tools."""

    def test_remember_narrative_tool_available(self):
        """Test that remember_narrative tool can be registered with FastMCP."""
        mcp = FastMCP("test-server")
        # Should not raise any exceptions during registration
        mcp.tool(remember_narrative)

        # If we get here, registration was successful
        assert True

    def test_remember_narrative_basic_functionality(self):
        """Test basic narrative storage functionality."""
        title = "Integration Test Story"
        paragraphs = [
            "This is the first paragraph of our test story.",
            "This is the second paragraph with more details.",
            "This is the final paragraph bringing it all together.",
        ]
        participants = ["Alpha", "Jeffery", "TestUser"]

        result = remember_narrative(title, paragraphs, participants)
        response_data = json.loads(result)

        # Verify successful storage
        assert response_data["success"] is True
        assert response_data["story"]["title"] == title
        assert response_data["story"]["participants"] == participants
        assert response_data["story"]["paragraph_count"] == len(paragraphs)
        assert response_data["processing"]["paragraphs_processed"] == len(paragraphs)

        # Verify story ID format
        story_id = response_data["story"]["story_id"]
        assert story_id.startswith("story_")
        assert len(story_id.split("_")) == 3  # story_timestamp_correlation

    def test_remember_narrative_with_all_options(self):
        """Test narrative storage with all optional parameters."""
        result = remember_narrative(
            title="Complete Integration Test",
            paragraphs=["Test paragraph 1", "Test paragraph 2"],
            participants=["Alpha", "Jeffery"],
            outcome="breakthrough",
            tags=["integration", "testing", "memory"],
            references=["story_123_abc", "story_456_def"],
        )

        response_data = json.loads(result)
        assert response_data["success"] is True

        story = response_data["story"]
        assert story["outcome"] == "breakthrough"
        assert story["tags"] == ["integration", "testing", "memory"]
        assert story["references"] == ["story_123_abc", "story_456_def"]

    def test_remember_narrative_validation_errors(self):
        """Test that validation errors are properly handled."""
        # Test empty title
        result = remember_narrative("", ["paragraph"], ["participant"])
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "Title cannot be empty" in response_data["error"]

        # Test empty paragraphs
        result = remember_narrative("Title", [], ["participant"])
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "At least one non-empty paragraph required" in response_data["error"]

        # Test empty participants
        result = remember_narrative("Title", ["paragraph"], [])
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "At least one participant required" in response_data["error"]

    def test_search_narratives_tool_available(self):
        """Test that search_narratives tool can be registered with FastMCP."""
        mcp = FastMCP("test-server")
        # Should not raise any exceptions during registration
        mcp.tool(search_narratives)

        # If we get here, registration was successful
        assert True

    def test_search_narratives_basic_functionality(self):
        """Test basic narrative search functionality."""
        result = search_narratives("test query")
        response_data = json.loads(result)

        # Verify successful search
        assert response_data["success"] is True
        assert response_data["search"]["query"] == "test query"
        assert response_data["search"]["search_type"] == "semantic"
        assert response_data["search"]["granularity"] == "story"
        assert response_data["search"]["limit"] == 10

        # Verify metadata
        metadata = response_data["metadata"]
        assert metadata["search_method"] == "vector_similarity"
        assert metadata["embedding_model"] == "dual_semantic_emotional"
        assert metadata["results_count"] == 0  # placeholder value

    def test_search_narratives_with_parameters(self):
        """Test search with various parameter combinations."""
        result = search_narratives(
            query="advanced search",
            search_type="both",
            granularity="paragraph",
            limit=5,
        )

        response_data = json.loads(result)
        assert response_data["success"] is True

        search = response_data["search"]
        assert search["query"] == "advanced search"
        assert search["search_type"] == "both"
        assert search["granularity"] == "paragraph"
        assert search["limit"] == 5

    def test_search_narratives_validation_errors(self):
        """Test search validation errors."""
        # Test empty query
        result = search_narratives("")
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "Query cannot be empty" in response_data["error"]

        # Test invalid search_type
        result = search_narratives("query", search_type="invalid")
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "search_type must be" in response_data["error"]

    def test_recall_narrative_tool_available(self):
        """Test that recall_narrative tool can be registered with FastMCP."""
        mcp = FastMCP("test-server")
        # Should not raise any exceptions during registration
        mcp.tool(recall_narrative)

        # If we get here, registration was successful
        assert True

    def test_recall_narrative_basic_functionality(self):
        """Test basic narrative recall functionality."""
        story_id = "story_1234567890_abc123"
        result = recall_narrative(story_id)
        response_data = json.loads(result)

        # Verify successful recall
        assert response_data["success"] is True
        assert response_data["story"]["story_id"] == story_id

        # Verify story structure
        story = response_data["story"]
        assert "title" in story
        assert "created_at" in story
        assert "participants" in story
        assert "paragraphs" in story
        assert "metadata" in story

        # Verify paragraphs structure
        paragraphs = story["paragraphs"]
        assert isinstance(paragraphs, list)
        for para in paragraphs:
            assert "order" in para
            assert "content" in para

    def test_recall_narrative_validation_errors(self):
        """Test recall validation errors."""
        # Test empty story_id
        result = recall_narrative("")
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "story_id cannot be empty" in response_data["error"]

        # Test invalid story_id format
        result = recall_narrative("invalid_format")
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "story_id must start with 'story_'" in response_data["error"]

    def test_browse_narrative_tool_available(self):
        """Test that browse_narrative tool can be registered with FastMCP."""
        mcp = FastMCP("test-server")
        # Should not raise any exceptions during registration
        mcp.tool(browse_narrative)

        # If we get here, registration was successful
        assert True

    def test_browse_narrative_basic_functionality(self):
        """Test basic narrative browsing functionality."""
        result = browse_narrative()
        response_data = json.loads(result)

        # Verify successful browse
        assert response_data["success"] is True

        # Verify browse_data structure
        browse_data = response_data["browse_data"]
        assert "stories" in browse_data
        assert "pagination" in browse_data
        assert "filters" in browse_data
        assert "metadata" in browse_data

        # Verify pagination
        pagination = browse_data["pagination"]
        assert pagination["limit"] == 10
        assert pagination["offset"] == 0
        assert pagination["total_count"] == 0  # placeholder

        # Verify metadata
        metadata = browse_data["metadata"]
        assert metadata["sort_order"] == "chronological_desc"
        assert metadata["storage_location"] == "hybrid_redis_memgraph"

    def test_browse_narrative_with_filters(self):
        """Test browsing with various filter parameters."""
        result = browse_narrative(
            limit=5,
            offset=10,
            since="7d",
            participants=["Alpha", "Jeffery"],
            tags=["important", "breakthrough"],
            outcome="resolution",
        )

        response_data = json.loads(result)
        assert response_data["success"] is True

        browse_data = response_data["browse_data"]

        # Verify pagination
        pagination = browse_data["pagination"]
        assert pagination["limit"] == 5
        assert pagination["offset"] == 10

        # Verify filters
        filters = browse_data["filters"]
        assert filters["since"] == "7d"
        assert filters["participants"] == ["Alpha", "Jeffery"]
        assert filters["tags"] == ["important", "breakthrough"]
        assert filters["outcome"] == "resolution"

    def test_browse_narrative_validation_errors(self):
        """Test browse validation errors."""
        # Test invalid limit
        result = browse_narrative(limit=0)
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "limit must be between" in response_data["error"]

        # Test invalid offset
        result = browse_narrative(offset=-1)
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "offset must be non-negative" in response_data["error"]

    def test_narrative_tools_correlation_id_consistency(self):
        """Test that all narrative tools generate proper correlation IDs."""
        # Test remember_narrative
        result = remember_narrative("Test", ["paragraph"], ["participant"])
        response_data = json.loads(result)
        correlation_id = response_data["correlation_id"]
        assert correlation_id.startswith("narrative_")

        # Test search_narratives
        result = search_narratives("query")
        response_data = json.loads(result)
        correlation_id = response_data["correlation_id"]
        assert correlation_id.startswith("search_narratives_")

        # Test recall_narrative
        result = recall_narrative("story_123_abc")
        response_data = json.loads(result)
        correlation_id = response_data["correlation_id"]
        assert correlation_id.startswith("recall_narrative_")

        # Test browse_narrative
        result = browse_narrative()
        response_data = json.loads(result)
        correlation_id = response_data["correlation_id"]
        assert correlation_id.startswith("browse_narrative_")

    def test_narrative_tools_error_handling(self):
        """Test that all narrative tools handle errors consistently."""
        # Each tool should return proper error structure
        tools_and_invalid_args = [
            (remember_narrative, ("", ["p"], ["participant"])),
            (search_narratives, ("",)),
            (recall_narrative, ("",)),
            (browse_narrative, {"limit": 0}),
        ]

        for tool_func, args in tools_and_invalid_args:
            if isinstance(args, dict):
                result = tool_func(**args)
            else:
                result = tool_func(*args)

            response_data = json.loads(result)
            assert response_data["success"] is False
            assert "error" in response_data
            assert "error_type" in response_data
            assert "correlation_id" in response_data

    def test_narrative_tools_json_response_consistency(self):
        """Test that all narrative tools return valid JSON responses."""
        # Test successful responses
        tools_and_valid_args = [
            (remember_narrative, ("Title", ["paragraph"], ["participant"])),
            (search_narratives, ("query",)),
            (recall_narrative, ("story_123_abc",)),
            (browse_narrative, ()),
        ]

        for tool_func, args in tools_and_valid_args:
            result = tool_func(*args)

            # Should be valid JSON
            response_data = json.loads(result)
            assert isinstance(response_data, dict)
            assert "success" in response_data
            assert "correlation_id" in response_data

            # Should be re-serializable
            json.dumps(response_data)

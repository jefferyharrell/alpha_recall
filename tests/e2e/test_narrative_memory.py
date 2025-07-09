"""Narrative memory tests for Alpha-Recall MCP server."""

import json

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_remember_narrative_tool(test_stack):
    """Test the remember_narrative tool via MCP interface."""
    async with Client(test_stack) as client:
        # Test narrative storage
        result = await client.call_tool(
            "remember_narrative",
            {
                "title": "E2E Test Story",
                "paragraphs": [
                    "This is the first paragraph of our E2E test story.",
                    "This is the second paragraph with more technical details.",
                    "This is the final paragraph bringing it all together.",
                ],
                "participants": ["Alpha", "E2E_TestUser"],
                "tags": ["testing", "e2e", "narrative"],
                "outcome": "successful",
                "references": ["story_test_ref_123"],
            },
        )

        # Parse the JSON response
        response_data = json.loads(result.content[0].text)

        # Debug: Print the response if it failed
        if not response_data.get("success", False):
            print(f"DEBUG: remember_narrative failed with response: {response_data}")

        # Verify the response structure
        assert "success" in response_data
        assert response_data["success"] is True
        assert "story" in response_data
        assert "processing" in response_data
        assert "correlation_id" in response_data

        # Verify story structure
        story = response_data["story"]
        assert "story_id" in story
        assert story["title"] == "E2E Test Story"
        assert story["participants"] == ["Alpha", "E2E_TestUser"]
        assert story["tags"] == ["testing", "e2e", "narrative"]
        assert story["outcome"] == "successful"
        assert story["references"] == ["story_test_ref_123"]
        assert story["paragraph_count"] == 3

        # Verify processing information
        processing = response_data["processing"]
        assert "paragraphs_processed" in processing
        assert "embeddings_generated" in processing
        assert processing["paragraphs_processed"] == 3
        assert processing["embeddings_generated"] > 0

        # Store the story_id for subsequent tests
        return story["story_id"]


@pytest.mark.asyncio
async def test_recall_narrative_tool(test_stack):
    """Test the recall_narrative tool via MCP interface."""
    async with Client(test_stack) as client:
        # First create a story to recall
        create_result = await client.call_tool(
            "remember_narrative",
            {
                "title": "E2E Recall Test Story",
                "paragraphs": ["This paragraph will be recalled in the E2E test."],
                "participants": ["Alpha", "E2E_RecallUser"],
            },
        )

        create_data = json.loads(create_result.content[0].text)
        assert create_data["success"] is True
        story_id = create_data["story"]["story_id"]

        # Now recall the story
        recall_result = await client.call_tool(
            "recall_narrative",
            {"story_id": story_id},
        )

        # Parse the JSON response
        response_data = json.loads(recall_result.content[0].text)

        # Verify successful recall
        assert "success" in response_data
        assert response_data["success"] is True
        assert "story" in response_data
        assert "correlation_id" in response_data

        # Verify story structure
        story = response_data["story"]
        assert story["story_id"] == story_id
        assert story["title"] == "E2E Recall Test Story"
        assert story["participants"] == ["Alpha", "E2E_RecallUser"]
        assert "paragraphs" in story
        assert "metadata" in story

        # Verify paragraphs structure
        paragraphs = story["paragraphs"]
        assert isinstance(paragraphs, list)
        assert len(paragraphs) == 1
        assert (
            paragraphs[0]["text"] == "This paragraph will be recalled in the E2E test."
        )
        assert paragraphs[0]["order"] == 0

        print(f"✅ Successfully recalled story: {story_id}")


@pytest.mark.asyncio
async def test_search_narratives_tool(test_stack):
    """Test the search_narratives tool via MCP interface."""
    async with Client(test_stack) as client:
        # First create a story to search for
        create_result = await client.call_tool(
            "remember_narrative",
            {
                "title": "E2E Search Test Story",
                "paragraphs": [
                    "This story contains unique searchable content for E2E testing.",
                    "The narrative search functionality should find this easily.",
                ],
                "participants": ["Alpha", "E2E_SearchUser"],
                "tags": ["searchable", "unique", "e2e"],
            },
        )

        create_data = json.loads(create_result.content[0].text)
        assert create_data["success"] is True

        # Now search for the story
        search_result = await client.call_tool(
            "search_narratives",
            {
                "query": "unique searchable content E2E testing",
                "search_type": "semantic",
                "granularity": "story",
                "limit": 10,
            },
        )

        # Parse the JSON response
        response_data = json.loads(search_result.content[0].text)

        # Verify successful search
        assert "success" in response_data
        assert response_data["success"] is True
        assert "results" in response_data
        assert "search" in response_data
        assert "correlation_id" in response_data

        # Verify search metadata
        search_meta = response_data["search"]
        assert search_meta["query"] == "unique searchable content E2E testing"
        assert search_meta["search_type"] == "semantic"
        assert search_meta["granularity"] == "story"
        assert search_meta["limit"] == 10

        # Verify results structure
        results = response_data["results"]
        assert isinstance(results, list)
        assert len(results) > 0  # Should find at least our created story

        # Verify result structure
        for result in results:
            assert "story_id" in result
            assert "title" in result
            assert "participants" in result
            assert "similarity_score" in result
            assert "search_type" in result
            assert 0 <= result["similarity_score"] <= 1

        print(f"✅ Search found {len(results)} stories")


@pytest.mark.asyncio
async def test_browse_narrative_tool(test_stack):
    """Test the browse_narrative tool via MCP interface."""
    async with Client(test_stack) as client:
        # Test basic browsing
        browse_result = await client.call_tool(
            "browse_narrative",
            {"limit": 5, "offset": 0},
        )

        # Parse the JSON response
        response_data = json.loads(browse_result.content[0].text)

        # Debug: Print the response if it failed
        if not response_data.get("success", False):
            print(f"DEBUG: browse_narrative failed with response: {response_data}")

        # Verify successful browse
        assert "success" in response_data
        assert response_data["success"] is True
        assert "browse_data" in response_data
        assert "correlation_id" in response_data

        # Verify browse_data structure
        browse_data = response_data["browse_data"]
        assert "stories" in browse_data
        assert "pagination" in browse_data
        assert "filters" in browse_data
        assert "metadata" in browse_data

        # Verify pagination
        pagination = browse_data["pagination"]
        assert pagination["limit"] == 5
        assert pagination["offset"] == 0
        assert "total_count" in pagination
        assert "has_more" in pagination
        assert isinstance(pagination["total_count"], int)
        assert pagination["total_count"] >= 0

        # Verify stories structure
        stories = browse_data["stories"]
        assert isinstance(stories, list)
        assert len(stories) <= 5  # Should respect limit

        # Verify story structure in browse results
        for story in stories:
            assert "story_id" in story
            assert "title" in story
            assert "participants" in story
            assert "created_at" in story
            assert "tags" in story
            assert "outcome" in story

        # Verify metadata
        metadata = browse_data["metadata"]
        assert metadata["sort_order"] == "chronological_desc"
        assert metadata["storage_location"] == "hybrid_redis_memgraph"

        print(
            f"✅ Browse found {len(stories)} stories (total: {pagination['total_count']})"
        )


@pytest.mark.asyncio
async def test_narrative_tools_integration(test_stack):
    """Test full narrative workflow: create -> search -> recall -> browse."""
    async with Client(test_stack) as client:
        # 1. Create a story
        create_result = await client.call_tool(
            "remember_narrative",
            {
                "title": "E2E Integration Test Workflow",
                "paragraphs": [
                    "This is a comprehensive integration test of the narrative memory system.",
                    "We're testing the full workflow from creation to retrieval.",
                    "This demonstrates the complete narrative memory capabilities.",
                ],
                "participants": ["Alpha", "E2E_IntegrationUser"],
                "tags": ["integration", "workflow", "comprehensive"],
                "outcome": "demonstration",
            },
        )

        create_data = json.loads(create_result.content[0].text)
        assert create_data["success"] is True
        story_id = create_data["story"]["story_id"]
        original_title = create_data["story"]["title"]

        # 2. Search for the story
        search_result = await client.call_tool(
            "search_narratives",
            {
                "query": "comprehensive integration test workflow",
                "search_type": "semantic",
                "limit": 5,
            },
        )

        search_data = json.loads(search_result.content[0].text)
        assert search_data["success"] is True
        assert len(search_data["results"]) > 0

        # Verify our story appears in search results
        found_story = None
        for result in search_data["results"]:
            if result["story_id"] == story_id:
                found_story = result
                break

        assert found_story is not None
        assert found_story["title"] == original_title

        # 3. Recall the specific story
        recall_result = await client.call_tool(
            "recall_narrative",
            {"story_id": story_id},
        )

        recall_data = json.loads(recall_result.content[0].text)
        assert recall_data["success"] is True
        assert recall_data["story"]["story_id"] == story_id
        assert recall_data["story"]["title"] == original_title

        # 4. Browse stories and verify our story appears
        browse_result = await client.call_tool(
            "browse_narrative",
            {"limit": 10},
        )

        browse_data = json.loads(browse_result.content[0].text)
        assert browse_data["success"] is True

        # Verify our story appears in browse results
        found_in_browse = False
        for story in browse_data["browse_data"]["stories"]:
            if story["story_id"] == story_id:
                found_in_browse = True
                assert story["title"] == original_title
                break

        assert (
            found_in_browse
            or browse_data["browse_data"]["pagination"]["total_count"] > 10
        )

        print(f"✅ Full workflow test completed successfully for story: {story_id}")


@pytest.mark.asyncio
async def test_narrative_error_handling(test_stack):
    """Test error handling in narrative memory tools."""
    async with Client(test_stack) as client:
        # Test recall with invalid story_id
        recall_result = await client.call_tool(
            "recall_narrative",
            {"story_id": "invalid_story_id"},
        )

        recall_data = json.loads(recall_result.content[0].text)
        assert recall_data["success"] is False
        assert "error" in recall_data

        # Test remember with invalid parameters
        remember_result = await client.call_tool(
            "remember_narrative",
            {
                "title": "",  # Empty title should fail
                "paragraphs": ["Some content"],
                "participants": ["Alpha"],
            },
        )

        remember_data = json.loads(remember_result.content[0].text)
        assert remember_data["success"] is False
        assert "error" in remember_data

        # Test search with invalid parameters
        search_result = await client.call_tool(
            "search_narratives",
            {
                "query": "",  # Empty query should fail
                "search_type": "semantic",
            },
        )

        search_data = json.loads(search_result.content[0].text)
        assert search_data["success"] is False
        assert "error" in search_data

        print("✅ Error handling tests completed successfully")

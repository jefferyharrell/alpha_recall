"""Unit tests for the conversational consolidation service."""

from unittest.mock import MagicMock, patch

import pytest
from src.alpha_recall.schemas.consolidation import ShortTermMemory
from src.alpha_recall.services.consolidation import (
    ConsolidationService,
    consolidation_service,
)


@pytest.fixture
def mock_redis_service():
    """Mock Redis service for testing."""
    mock_service = MagicMock()
    mock_client = MagicMock()
    mock_service.client = mock_client

    # Mock memory retrieval
    mock_client.zrevrangebyscore.return_value = [
        b"stm_1234567890123_abc123",
        b"stm_1234567890124_def456",
    ]
    mock_client.hmget.side_effect = [
        [b"Test memory content 1", b"2025-07-10T12:00:00.000000+00:00"],
        [b"Test memory content 2", b"2025-07-10T12:01:00.000000+00:00"],
    ]

    return mock_service


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama response for testing."""
    return {
        "message": {
            "content": "Hey there, friend! I'm reflecting on the recent memories and they show an interesting pattern of testing and development work. The team has been making good progress on the consolidation system with positive results."
        }
    }


@pytest.fixture
def mock_template_loader():
    """Mock template loader for testing."""
    mock_loader = MagicMock()
    mock_loader.render_template.return_value = (
        "Mock consolidation prompt with {{ memory_count }} memories"
    )
    return mock_loader


class TestConsolidationService:
    """Test cases for the ConsolidationService class."""

    def test_service_initialization(self):
        """Test that the consolidation service initializes correctly."""
        service = ConsolidationService()
        assert service is not None

    def test_global_service_instance(self):
        """Test that the global consolidation service instance exists."""
        assert consolidation_service is not None
        assert isinstance(consolidation_service, ConsolidationService)

    @pytest.mark.asyncio
    @patch("src.alpha_recall.services.consolidation.get_redis_service")
    @patch("src.alpha_recall.services.consolidation.template_loader")
    @patch("src.alpha_recall.services.consolidation.ollama.chat")
    async def test_consolidate_shortterm_memories_success(
        self,
        mock_ollama_chat,
        mock_template_loader,
        mock_get_redis_service,
        mock_redis_service,
        mock_ollama_response,
    ):
        """Test successful consolidation with conversational approach."""
        # Setup mocks
        mock_get_redis_service.return_value = mock_redis_service
        mock_template_loader.render_template.return_value = "Mock prompt"
        mock_ollama_chat.return_value = mock_ollama_response

        service = ConsolidationService()
        result = await service.consolidate_shortterm_memories(
            time_window="6h", model_name="llama3.2:1b", temperature=0.0
        )

        # Verify response structure
        assert result["success"] is True
        assert "narrative" in result
        assert "metadata" in result
        assert "correlation_id" in result
        assert "tool_metadata" in result

        # Verify metadata content
        metadata = result["metadata"]
        assert metadata["model_name"] == "llama3.2:1b"
        assert metadata["temperature"] == 0.0
        assert metadata["approach"] == "conversational_reflection"
        assert "processing_time_ms" in metadata
        assert "input_memories_count" in metadata

        # Verify tool metadata
        tool_metadata = result["tool_metadata"]
        assert tool_metadata["tool_name"] == "consolidate_shortterm"
        assert tool_metadata["tool_version"] == "2.0"
        assert tool_metadata["systematic_evaluation"] is True

    @patch("src.alpha_recall.services.consolidation.get_redis_service")
    async def test_consolidate_no_memories_found(
        self, mock_get_redis_service, mock_redis_service
    ):
        """Test consolidation when no memories are found."""
        # Setup mock to return no memories
        mock_redis_service.client.zrevrangebyscore.return_value = []
        mock_get_redis_service.return_value = mock_redis_service

        service = ConsolidationService()
        result = await service.consolidate_shortterm_memories(time_window="1h")

        # Verify empty response
        assert result["success"] is True
        assert (
            result["narrative"]
            == "No recent memories found to reflect on. It's been quiet lately."
        )
        assert result["metadata"]["input_memories_count"] == 0
        assert result["metadata"]["reason"] == "no_memories_found"
        assert result["metadata"]["approach"] == "conversational_reflection"

    @patch("src.alpha_recall.services.consolidation.get_redis_service")
    async def test_consolidate_redis_error(self, mock_get_redis_service):
        """Test consolidation when Redis fails."""
        # Setup mock to raise exception
        mock_get_redis_service.side_effect = Exception("Redis connection failed")

        service = ConsolidationService()
        result = await service.consolidate_shortterm_memories()

        # Verify error response
        assert result["success"] is False
        assert "error" in result
        assert "Redis connection failed" in result["error"]

    @patch("src.alpha_recall.services.consolidation.get_redis_service")
    @patch("src.alpha_recall.services.consolidation.ollama.chat")
    async def test_consolidate_ollama_error(
        self, mock_ollama_chat, mock_get_redis_service, mock_redis_service
    ):
        """Test consolidation when Ollama fails."""
        # Setup mocks
        mock_get_redis_service.return_value = mock_redis_service
        mock_ollama_chat.side_effect = Exception("Ollama model not available")

        service = ConsolidationService()
        result = await service.consolidate_shortterm_memories()

        # Verify error response
        assert result["success"] is False
        assert "error" in result
        assert "Ollama model not available" in result["error"]

    @patch("src.alpha_recall.services.consolidation.get_redis_service")
    @patch("src.alpha_recall.services.consolidation.template_loader")
    @patch("src.alpha_recall.services.consolidation.ollama.chat")
    async def test_time_window_parsing(
        self,
        mock_ollama_chat,
        mock_template_loader,
        mock_get_redis_service,
        mock_redis_service,
        mock_ollama_response,
    ):
        """Test that different time window formats are handled correctly."""
        # Setup mocks
        mock_get_redis_service.return_value = mock_redis_service
        mock_template_loader.render_template.return_value = "Mock prompt"
        mock_ollama_chat.return_value = mock_ollama_response

        service = ConsolidationService()

        # Test different time window formats
        test_windows = ["1h", "24h", "7d", "30m"]

        for window in test_windows:
            result = await service.consolidate_shortterm_memories(time_window=window)
            assert result["success"] is True
            assert result["metadata"]["time_window"] == window

    @patch("src.alpha_recall.services.consolidation.get_redis_service")
    @patch("src.alpha_recall.services.consolidation.template_loader")
    @patch("src.alpha_recall.services.consolidation.ollama.chat")
    async def test_model_override(
        self,
        mock_ollama_chat,
        mock_template_loader,
        mock_get_redis_service,
        mock_redis_service,
        mock_ollama_response,
    ):
        """Test that model name can be overridden for testing."""
        # Setup mocks
        mock_get_redis_service.return_value = mock_redis_service
        mock_template_loader.render_template.return_value = "Mock prompt"
        mock_ollama_chat.return_value = mock_ollama_response

        service = ConsolidationService()
        result = await service.consolidate_shortterm_memories(
            model_name="llama3.1:8b", temperature=0.5
        )

        assert result["success"] is True
        assert result["metadata"]["model_name"] == "llama3.1:8b"
        assert result["metadata"]["temperature"] == 0.5

        # Verify Ollama was called with correct parameters
        mock_ollama_chat.assert_called_once()
        call_args = mock_ollama_chat.call_args
        assert call_args[1]["model"] == "llama3.1:8b"
        assert call_args[1]["options"]["temperature"] == 0.5

    def test_prompt_template_method(self):
        """Test the conversational prompt template method."""
        service = ConsolidationService()

        # Create test memories
        test_memories = [
            ShortTermMemory(
                content="Test memory 1", timestamp="2025-07-10T12:00:00.000000+00:00"
            ),
            ShortTermMemory(
                content="Test memory 2", timestamp="2025-07-10T12:01:00.000000+00:00"
            ),
        ]

        with patch(
            "src.alpha_recall.services.consolidation.template_loader"
        ) as mock_loader:
            mock_loader.render_template.return_value = "Rendered prompt"

            result = service._get_conversational_prompt(test_memories, "6h")

            assert result == "Rendered prompt"
            mock_loader.render_template.assert_called_once_with(
                "memory_consolidation.md.j2",
                {
                    "memories": [
                        {
                            "content": "Test memory 1",
                            "timestamp": "2025-07-10T12:00:00.000000+00:00",
                        },
                        {
                            "content": "Test memory 2",
                            "timestamp": "2025-07-10T12:01:00.000000+00:00",
                        },
                    ],
                    "memory_count": 2,
                    "time_window": "6h",
                },
            )

    def test_prompt_template_fallback(self):
        """Test that prompt template has a fallback when template loading fails."""
        service = ConsolidationService()

        test_memories = [
            ShortTermMemory(
                content="Test memory", timestamp="2025-07-10T12:00:00.000000+00:00"
            )
        ]

        with patch(
            "src.alpha_recall.services.consolidation.template_loader"
        ) as mock_loader:
            mock_loader.render_template.side_effect = Exception("Template not found")

            result = service._get_conversational_prompt(test_memories, "1h")

            # Should return fallback prompt
            assert "Hey there!" in result
            assert "1 recent memories" in result
            assert "Test memory" in result
            assert "What's the story here?" in result

    @patch("src.alpha_recall.services.consolidation.get_redis_service")
    @patch("src.alpha_recall.services.consolidation.template_loader")
    @patch("src.alpha_recall.services.consolidation.ollama.chat")
    async def test_correlation_id_generation(
        self,
        mock_ollama_chat,
        mock_template_loader,
        mock_get_redis_service,
        mock_redis_service,
        mock_ollama_response,
    ):
        """Test that correlation IDs are generated and tracked."""
        # Setup mocks
        mock_get_redis_service.return_value = mock_redis_service
        mock_template_loader.render_template.return_value = "Mock prompt"
        mock_ollama_chat.return_value = mock_ollama_response

        service = ConsolidationService()
        result = await service.consolidate_shortterm_memories()

        # Verify correlation ID is present and follows expected format
        assert "correlation_id" in result
        correlation_id = result["correlation_id"]
        assert correlation_id.startswith("consolidate_")
        assert len(correlation_id.split("_")) == 2  # consolidate_<uuid>

    @patch("src.alpha_recall.services.consolidation.get_redis_service")
    @patch("src.alpha_recall.services.consolidation.template_loader")
    @patch("src.alpha_recall.services.consolidation.ollama.chat")
    async def test_response_length_tracking(
        self,
        mock_ollama_chat,
        mock_template_loader,
        mock_get_redis_service,
        mock_redis_service,
    ):
        """Test that response length is tracked in metadata."""
        # Setup mocks with specific response length
        test_response = "This is a test response that has exactly 50 chars."
        mock_get_redis_service.return_value = mock_redis_service
        mock_template_loader.render_template.return_value = "Mock prompt"
        mock_ollama_chat.return_value = {"message": {"content": test_response}}

        service = ConsolidationService()
        result = await service.consolidate_shortterm_memories()

        assert result["success"] is True
        assert result["metadata"]["response_length"] == len(test_response)
        assert result["narrative"] == test_response

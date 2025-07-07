"""Unit tests for EmbeddingService."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.services.embedding import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService."""

    @patch("alpha_recall.services.embedding.SentenceTransformer")
    @patch("alpha_recall.services.embedding.settings")
    def test_init_with_explicit_device(self, mock_settings, mock_sentence_transformer):
        """Test EmbeddingService initialization with explicit device configuration."""
        # Mock settings
        mock_settings.inference_device = "cuda"
        mock_settings.embedding_model = "test-semantic-model"
        mock_settings.emotional_embedding_model = "test-emotional-model"

        # Mock SentenceTransformer
        mock_semantic_model = MagicMock()
        mock_emotional_model = MagicMock()
        mock_sentence_transformer.side_effect = [
            mock_semantic_model,
            mock_emotional_model,
        ]

        # Initialize service
        service = EmbeddingService()

        # Verify device was set correctly
        assert service.device == "cuda"

        # Verify models were initialized with device
        assert mock_sentence_transformer.call_count == 2
        mock_sentence_transformer.assert_any_call("test-semantic-model", device="cuda")
        mock_sentence_transformer.assert_any_call("test-emotional-model", device="cuda")

    @patch("alpha_recall.services.embedding.SentenceTransformer")
    @patch("alpha_recall.services.embedding.settings")
    def test_init_with_auto_detection(self, mock_settings, mock_sentence_transformer):
        """Test EmbeddingService initialization with auto device detection."""
        # Mock settings
        mock_settings.inference_device = None
        mock_settings.embedding_model = "test-semantic-model"
        mock_settings.emotional_embedding_model = "test-emotional-model"

        # Mock SentenceTransformer
        mock_semantic_model = MagicMock()
        mock_emotional_model = MagicMock()
        mock_sentence_transformer.side_effect = [
            mock_semantic_model,
            mock_emotional_model,
        ]

        # Initialize service
        service = EmbeddingService()

        # Verify device was set to None (auto-detection)
        assert service.device is None

        # Verify models were initialized without device parameter
        assert mock_sentence_transformer.call_count == 2
        mock_sentence_transformer.assert_any_call("test-semantic-model")
        mock_sentence_transformer.assert_any_call("test-emotional-model")

    def test_encode_semantic_single_string(self):
        """Test semantic encoding of a single string."""
        # Mock the service
        service = EmbeddingService.__new__(EmbeddingService)
        service.semantic_model = MagicMock()

        # Mock encode method to return numpy array
        mock_embedding = np.array([0.1, 0.2, 0.3])
        service.semantic_model.encode.return_value = mock_embedding

        # Test encoding
        result = service.encode_semantic("test text")

        # Verify result
        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]
        service.semantic_model.encode.assert_called_once_with(
            "test text", convert_to_tensor=False
        )

    def test_encode_semantic_list_of_strings(self):
        """Test semantic encoding of a list of strings."""
        # Mock the service
        service = EmbeddingService.__new__(EmbeddingService)
        service.semantic_model = MagicMock()

        # Mock encode method to return numpy array of arrays
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        service.semantic_model.encode.return_value = mock_embeddings

        # Test encoding
        result = service.encode_semantic(["text1", "text2"])

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        service.semantic_model.encode.assert_called_once_with(
            ["text1", "text2"], convert_to_tensor=False
        )

    def test_encode_emotional_single_string(self):
        """Test emotional encoding of a single string."""
        # Mock the service
        service = EmbeddingService.__new__(EmbeddingService)
        service.emotional_model = MagicMock()

        # Mock encode method to return numpy array
        mock_embedding = np.array([0.5, 0.6, 0.7])
        service.emotional_model.encode.return_value = mock_embedding

        # Test encoding
        result = service.encode_emotional("test text")

        # Verify result
        assert isinstance(result, list)
        assert result == [0.5, 0.6, 0.7]
        service.emotional_model.encode.assert_called_once_with(
            "test text", convert_to_tensor=False
        )

    def test_encode_emotional_list_of_strings(self):
        """Test emotional encoding of a list of strings."""
        # Mock the service
        service = EmbeddingService.__new__(EmbeddingService)
        service.emotional_model = MagicMock()

        # Mock encode method to return numpy array of arrays
        mock_embeddings = np.array([[0.5, 0.6], [0.7, 0.8]])
        service.emotional_model.encode.return_value = mock_embeddings

        # Test encoding
        result = service.encode_emotional(["text1", "text2"])

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.5, 0.6]
        assert result[1] == [0.7, 0.8]
        service.emotional_model.encode.assert_called_once_with(
            ["text1", "text2"], convert_to_tensor=False
        )

    @patch("alpha_recall.services.embedding.SentenceTransformer")
    @patch("alpha_recall.services.embedding.settings")
    def test_determine_device_explicit_config(
        self, mock_settings, mock_sentence_transformer
    ):
        """Test device determination with explicit configuration."""
        # Mock settings
        mock_settings.inference_device = "mps"
        mock_settings.embedding_model = "test-model"
        mock_settings.emotional_embedding_model = "test-model"

        # Mock SentenceTransformer
        mock_sentence_transformer.return_value = MagicMock()

        # Initialize service
        service = EmbeddingService()

        # Test device determination
        device = service._determine_device()

        # Verify explicit device is returned
        assert device == "mps"

    @patch("alpha_recall.services.embedding.SentenceTransformer")
    @patch("alpha_recall.services.embedding.settings")
    def test_determine_device_auto_detection(
        self, mock_settings, mock_sentence_transformer
    ):
        """Test device determination with auto-detection."""
        # Mock settings
        mock_settings.inference_device = None
        mock_settings.embedding_model = "test-model"
        mock_settings.emotional_embedding_model = "test-model"

        # Mock SentenceTransformer
        mock_sentence_transformer.return_value = MagicMock()

        # Initialize service
        service = EmbeddingService()

        # Test device determination
        device = service._determine_device()

        # Verify None is returned for auto-detection
        assert device is None

    def test_encoding_type_consistency(self):
        """Test that encoding methods return consistent types."""
        # Mock the service
        service = EmbeddingService.__new__(EmbeddingService)
        service.semantic_model = MagicMock()
        service.emotional_model = MagicMock()

        # Mock encode methods
        service.semantic_model.encode.return_value = np.array([1.0, 2.0, 3.0])
        service.emotional_model.encode.return_value = np.array([4.0, 5.0, 6.0])

        # Test both methods
        semantic_result = service.encode_semantic("test")
        emotional_result = service.encode_emotional("test")

        # Verify both return lists of floats
        assert isinstance(semantic_result, list)
        assert isinstance(emotional_result, list)
        assert all(isinstance(x, float) for x in semantic_result)
        assert all(isinstance(x, float) for x in emotional_result)

    def test_encoding_empty_input_handling(self):
        """Test how encoding methods handle edge cases."""
        # Mock the service
        service = EmbeddingService.__new__(EmbeddingService)
        service.semantic_model = MagicMock()

        # Mock encode method for empty string
        service.semantic_model.encode.return_value = np.array([])

        # Test encoding empty string
        result = service.encode_semantic("")

        # Verify it doesn't crash and returns a list
        assert isinstance(result, list)
        service.semantic_model.encode.assert_called_once_with(
            "", convert_to_tensor=False
        )

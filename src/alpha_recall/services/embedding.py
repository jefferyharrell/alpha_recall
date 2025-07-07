"""Embedding service for Alpha-Recall semantic and emotional encoding."""

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..logging import get_logger

logger = get_logger("services.embedding")


class EmbeddingService:
    """Service for generating semantic and emotional embeddings.

    Eagerly loads both models at initialization and provides clean
    encode methods for different embedding types.
    """

    def __init__(self):
        """Initialize the embedding service with eager model loading."""
        logger.info("Initializing EmbeddingService with eager model loading")

        # Determine device to use
        self.device = self._determine_device()
        if self.device is not None:
            logger.info(f"Using explicitly configured device: {self.device}")
        else:
            logger.info("Using sentence-transformers auto-detected device")

        # Eagerly load both models
        logger.info(f"Loading semantic model: {settings.embedding_model}")
        if self.device is not None:
            self.semantic_model = SentenceTransformer(
                settings.embedding_model, device=self.device
            )
        else:
            self.semantic_model = SentenceTransformer(settings.embedding_model)

        logger.info(f"Loading emotional model: {settings.emotional_embedding_model}")
        if self.device is not None:
            self.emotional_model = SentenceTransformer(
                settings.emotional_embedding_model, device=self.device
            )
        else:
            self.emotional_model = SentenceTransformer(
                settings.emotional_embedding_model
            )

        logger.info("EmbeddingService initialization complete")

    def _determine_device(self) -> str | None:
        """Determine the device to use for inference.

        Returns:
            Device string if explicitly configured, None to let sentence-transformers auto-detect
        """
        # If explicitly configured, use that
        if settings.inference_device is not None:
            logger.debug(
                f"Using explicitly configured device: {settings.inference_device}"
            )
            return settings.inference_device

        # Otherwise, let sentence-transformers auto-detect the best device
        # This is smart about CUDA availability, MPS on Apple Silicon, etc.
        logger.debug("Letting sentence-transformers auto-detect device")
        return None

    def encode_semantic(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Encode text for semantic similarity matching.

        Args:
            text: Single text string or list of strings to encode

        Returns:
            Single embedding vector or list of embedding vectors
        """
        logger.debug(
            f"Encoding semantic embedding for {len(text) if isinstance(text, list) else 1} text(s)"
        )

        # sentence-transformers returns numpy arrays, convert to Python lists
        embeddings = self.semantic_model.encode(text, convert_to_tensor=False)

        if isinstance(text, str):
            # Single text input -> return single embedding as list
            embedding_array: np.ndarray = embeddings  # type: ignore
            return embedding_array.tolist()
        else:
            # List input -> return list of embeddings
            embeddings_array: np.ndarray = embeddings  # type: ignore
            return [embedding.tolist() for embedding in embeddings_array]

    def encode_emotional(
        self, text: str | list[str]
    ) -> list[float] | list[list[float]]:
        """Encode text for emotional similarity matching.

        Args:
            text: Single text string or list of strings to encode

        Returns:
            Single embedding vector or list of embedding vectors
        """
        logger.debug(
            f"Encoding emotional embedding for {len(text) if isinstance(text, list) else 1} text(s)"
        )

        # sentence-transformers returns numpy arrays, convert to Python lists
        embeddings = self.emotional_model.encode(text, convert_to_tensor=False)

        if isinstance(text, str):
            # Single text input -> return single embedding as list
            embedding_array: np.ndarray = embeddings  # type: ignore
            return embedding_array.tolist()
        else:
            # List input -> return list of embeddings
            embeddings_array: np.ndarray = embeddings  # type: ignore
            return [embedding.tolist() for embedding in embeddings_array]


# Global embedding service instance
embedding_service = EmbeddingService()

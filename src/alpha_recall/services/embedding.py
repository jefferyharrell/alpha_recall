"""Embedding service for Alpha-Recall semantic and emotional encoding."""

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..logging import get_logger
from ..utils.correlation import (
    create_child_correlation_id,
    set_correlation_id,
)

logger = get_logger("services.embedding")


class EmbeddingService:
    """Service for generating semantic and emotional embeddings.

    Uses lazy loading - models are loaded on first use and cached thereafter.
    """

    def __init__(self):
        """Initialize the embedding service with lazy loading."""
        logger.info("Initializing EmbeddingService with lazy loading")

        # Determine device to use
        self.device = self._determine_device()

        # Models will be loaded on first use
        self.semantic_model = None
        self.emotional_model = None

        logger.info("EmbeddingService initialized - models will load on first use")

    def _load_semantic_model(self):
        """Load semantic model on first use."""
        if self.semantic_model is None:
            import time

            start = time.perf_counter()
            logger.info(
                "Loading semantic model on first use",
                model=settings.semantic_embedding_model,
            )

            if self.device is not None:
                self.semantic_model = SentenceTransformer(
                    settings.semantic_embedding_model, device=self.device
                )
            else:
                self.semantic_model = SentenceTransformer(
                    settings.semantic_embedding_model
                )

            load_time_ms = (time.perf_counter() - start) * 1000
            logger.info("Semantic model loaded", load_time_ms=round(load_time_ms, 2))

        return self.semantic_model

    def _load_emotional_model(self):
        """Load emotional model on first use."""
        if self.emotional_model is None:
            import time

            start = time.perf_counter()
            logger.info(
                "Loading emotional model on first use",
                model=settings.emotional_embedding_model,
            )

            if self.device is not None:
                self.emotional_model = SentenceTransformer(
                    settings.emotional_embedding_model, device=self.device
                )
            else:
                self.emotional_model = SentenceTransformer(
                    settings.emotional_embedding_model
                )

            load_time_ms = (time.perf_counter() - start) * 1000
            logger.info("Emotional model loaded", load_time_ms=round(load_time_ms, 2))

        return self.emotional_model

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
        import time

        # Create child correlation ID for this encoding operation
        child_corr_id = create_child_correlation_id("semantic_encode")
        set_correlation_id(child_corr_id)

        start_time = time.perf_counter()
        is_batch = isinstance(text, list)
        text_count = len(text) if is_batch else 1

        # Calculate total character count for throughput metrics
        if is_batch:
            total_chars = sum(len(t) for t in text)
            avg_length = total_chars / len(text)
        else:
            total_chars = len(text)
            avg_length = total_chars

        # Load model on first use
        model = self._load_semantic_model()

        logger.debug(
            "Starting semantic encoding",
            text_count=text_count,
            is_batch=is_batch,
            total_characters=total_chars,
            avg_text_length=round(avg_length, 2),
            model_device=str(model.device),
            operation="semantic_encode",
        )

        # sentence-transformers returns numpy arrays, convert to Python lists
        embeddings = model.encode(text, convert_to_tensor=False)

        encode_time_ms = (time.perf_counter() - start_time) * 1000

        if isinstance(text, str):
            # Single text input -> return single embedding as list
            embedding_array: np.ndarray = embeddings  # type: ignore
            result = embedding_array.tolist()

            logger.debug(
                "Semantic encoding completed",
                text_count=1,
                is_batch=False,
                encoding_time_ms=round(encode_time_ms, 2),
                chars_per_second=(
                    round(total_chars / (encode_time_ms / 1000), 2)
                    if encode_time_ms > 0
                    else 0
                ),
                embedding_dimensions=len(result),
                model_type="semantic",
                operation="semantic_encode",
            )
            return result
        else:
            # List input -> return list of embeddings
            embeddings_array: np.ndarray = embeddings  # type: ignore
            result = [embedding.tolist() for embedding in embeddings_array]

            logger.debug(
                "Semantic batch encoding completed",
                text_count=len(result),
                is_batch=True,
                encoding_time_ms=round(encode_time_ms, 2),
                chars_per_second=(
                    round(total_chars / (encode_time_ms / 1000), 2)
                    if encode_time_ms > 0
                    else 0
                ),
                texts_per_second=(
                    round(text_count / (encode_time_ms / 1000), 2)
                    if encode_time_ms > 0
                    else 0
                ),
                embedding_dimensions=len(result[0]) if result else 0,
                model_type="semantic",
                operation="semantic_encode",
            )
            return result

    def encode_emotional(
        self, text: str | list[str]
    ) -> list[float] | list[list[float]]:
        """Encode text for emotional similarity matching.

        Args:
            text: Single text string or list of strings to encode

        Returns:
            Single embedding vector or list of embedding vectors
        """
        import time

        # Create child correlation ID for this encoding operation
        child_corr_id = create_child_correlation_id("emotional_encode")
        set_correlation_id(child_corr_id)

        start_time = time.perf_counter()
        is_batch = isinstance(text, list)
        text_count = len(text) if is_batch else 1

        # Calculate total character count for throughput metrics
        if is_batch:
            total_chars = sum(len(t) for t in text)
            avg_length = total_chars / len(text)
        else:
            total_chars = len(text)
            avg_length = total_chars

        # Load model on first use
        model = self._load_emotional_model()

        logger.debug(
            "Starting emotional encoding",
            text_count=text_count,
            is_batch=is_batch,
            total_characters=total_chars,
            avg_text_length=round(avg_length, 2),
            model_device=str(model.device),
            operation="emotional_encode",
        )

        # sentence-transformers returns numpy arrays, convert to Python lists
        embeddings = model.encode(text, convert_to_tensor=False)

        encode_time_ms = (time.perf_counter() - start_time) * 1000

        if isinstance(text, str):
            # Single text input -> return single embedding as list
            embedding_array: np.ndarray = embeddings  # type: ignore
            result = embedding_array.tolist()

            logger.debug(
                "Emotional encoding completed",
                text_count=1,
                is_batch=False,
                encoding_time_ms=round(encode_time_ms, 2),
                chars_per_second=(
                    round(total_chars / (encode_time_ms / 1000), 2)
                    if encode_time_ms > 0
                    else 0
                ),
                embedding_dimensions=len(result),
                model_type="emotional",
                operation="emotional_encode",
            )
            return result
        else:
            # List input -> return list of embeddings
            embeddings_array: np.ndarray = embeddings  # type: ignore
            result = [embedding.tolist() for embedding in embeddings_array]

            logger.debug(
                "Emotional batch encoding completed",
                text_count=len(result),
                is_batch=True,
                encoding_time_ms=round(encode_time_ms, 2),
                chars_per_second=(
                    round(total_chars / (encode_time_ms / 1000), 2)
                    if encode_time_ms > 0
                    else 0
                ),
                texts_per_second=(
                    round(text_count / (encode_time_ms / 1000), 2)
                    if encode_time_ms > 0
                    else 0
                ),
                embedding_dimensions=len(result[0]) if result else 0,
                model_type="emotional",
                operation="emotional_encode",
            )
            return result


# Global embedding service instance
embedding_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """Get the global EmbeddingService instance."""
    return embedding_service

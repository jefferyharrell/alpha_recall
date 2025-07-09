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

    Eagerly loads both models at initialization and provides clean
    encode methods for different embedding types.
    """

    def __init__(self):
        """Initialize the embedding service with eager model loading."""
        import time

        start_time = time.perf_counter()
        logger.info("Initializing EmbeddingService with eager model loading")

        # Determine device to use
        device_start = time.perf_counter()
        self.device = self._determine_device()
        device_time_ms = (time.perf_counter() - device_start) * 1000

        if self.device is not None:
            logger.info(
                "Device configuration applied",
                device=self.device,
                device_detection_time_ms=round(device_time_ms, 2),
                configuration_source="explicit",
            )
        else:
            logger.info(
                "Device auto-detection completed",
                device_detection_time_ms=round(device_time_ms, 2),
                configuration_source="auto_detect",
            )

        # Eagerly load semantic model
        semantic_start = time.perf_counter()
        logger.info(
            "Loading semantic embedding model",
            model_name=settings.semantic_embedding_model,
            device=self.device or "auto_detected",
        )

        if self.device is not None:
            self.semantic_model = SentenceTransformer(
                settings.semantic_embedding_model, device=self.device
            )
        else:
            self.semantic_model = SentenceTransformer(settings.semantic_embedding_model)

        semantic_time_ms = (time.perf_counter() - semantic_start) * 1000

        # Get model info after loading
        semantic_device = str(self.semantic_model.device)
        model_dims = self.semantic_model.get_sentence_embedding_dimension()

        logger.info(
            "Semantic model loaded successfully",
            model_name=settings.semantic_embedding_model,
            device=semantic_device,
            dimensions=model_dims,
            load_time_ms=round(semantic_time_ms, 2),
            model_type="semantic",
        )

        # Eagerly load emotional model
        emotional_start = time.perf_counter()
        logger.info(
            "Loading emotional embedding model",
            model_name=settings.emotional_embedding_model,
            device=self.device or "auto_detected",
        )

        if self.device is not None:
            self.emotional_model = SentenceTransformer(
                settings.emotional_embedding_model, device=self.device
            )
        else:
            self.emotional_model = SentenceTransformer(
                settings.emotional_embedding_model
            )

        emotional_time_ms = (time.perf_counter() - emotional_start) * 1000

        # Get emotional model info
        emotional_device = str(self.emotional_model.device)
        emotional_dims = self.emotional_model.get_sentence_embedding_dimension()

        logger.info(
            "Emotional model loaded successfully",
            model_name=settings.emotional_embedding_model,
            device=emotional_device,
            dimensions=emotional_dims,
            load_time_ms=round(emotional_time_ms, 2),
            model_type="emotional",
        )

        total_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "EmbeddingService initialization complete",
            total_models_loaded=2,
            semantic_dimensions=model_dims,
            emotional_dimensions=emotional_dims,
            total_init_time_ms=round(total_time_ms, 2),
            semantic_load_time_ms=round(semantic_time_ms, 2),
            emotional_load_time_ms=round(emotional_time_ms, 2),
            device_used=semantic_device,
        )

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

        logger.debug(
            "Starting semantic encoding",
            text_count=text_count,
            is_batch=is_batch,
            total_characters=total_chars,
            avg_text_length=round(avg_length, 2),
            model_device=str(self.semantic_model.device),
            operation="semantic_encode",
        )

        # sentence-transformers returns numpy arrays, convert to Python lists
        embeddings = self.semantic_model.encode(text, convert_to_tensor=False)

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

        logger.debug(
            "Starting emotional encoding",
            text_count=text_count,
            is_batch=is_batch,
            total_characters=total_chars,
            avg_text_length=round(avg_length, 2),
            model_device=str(self.emotional_model.device),
            operation="emotional_encode",
        )

        # sentence-transformers returns numpy arrays, convert to Python lists
        embeddings = self.emotional_model.encode(text, convert_to_tensor=False)

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

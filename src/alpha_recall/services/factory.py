"""Service factory for Alpha-Recall v2.0."""

from ..config import AlphaRecallSettings
from .embedding import EmbeddingService
from .narrative import NarrativeService

# Global service instances (singleton pattern)
_embedding_service: EmbeddingService | None = None
_narrative_service: NarrativeService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the shared embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_narrative_service() -> NarrativeService:
    """Get or create the shared narrative service instance."""
    global _narrative_service
    if _narrative_service is None:
        settings = AlphaRecallSettings()
        embedding_service = get_embedding_service()
        _narrative_service = NarrativeService(embedding_service, settings)
    return _narrative_service


async def close_services():
    """Close all service instances."""
    global _embedding_service, _narrative_service

    if _narrative_service:
        await _narrative_service.close()
        _narrative_service = None

    if _embedding_service:
        # EmbeddingService doesn't have close method currently, but we'll reset it
        _embedding_service = None

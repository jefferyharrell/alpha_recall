"""Service factory for Alpha-Recall v2.0."""

from ..config import AlphaRecallSettings
from .embedding import EmbeddingService
from .narrative import NarrativeService
from .redis_context import RedisContextService
from .redis_identity import RedisIdentityService
from .redis_memory import RedisMemoryService

# Global service instances (singleton pattern)
_embedding_service: EmbeddingService | None = None
_narrative_service: NarrativeService | None = None
_redis_memory_service: RedisMemoryService | None = None
_redis_identity_service: RedisIdentityService | None = None
_redis_context_service: RedisContextService | None = None


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


def get_redis_memory_service() -> RedisMemoryService:
    """Get or create the shared Redis memory service instance."""
    global _redis_memory_service
    if _redis_memory_service is None:
        _redis_memory_service = RedisMemoryService()
    return _redis_memory_service


def get_redis_identity_service() -> RedisIdentityService:
    """Get or create the shared Redis identity service instance."""
    global _redis_identity_service
    if _redis_identity_service is None:
        _redis_identity_service = RedisIdentityService()
    return _redis_identity_service


def get_redis_context_service() -> RedisContextService:
    """Get or create the shared Redis context service instance."""
    global _redis_context_service
    if _redis_context_service is None:
        _redis_context_service = RedisContextService()
    return _redis_context_service


async def close_services():
    """Close all service instances."""
    global _embedding_service, _narrative_service
    global _redis_memory_service, _redis_identity_service, _redis_context_service

    if _narrative_service:
        await _narrative_service.close()
        _narrative_service = None

    if _embedding_service:
        # EmbeddingService doesn't have close method currently, but we'll reset it
        _embedding_service = None

    # Reset Redis service instances
    _redis_memory_service = None
    _redis_identity_service = None
    _redis_context_service = None

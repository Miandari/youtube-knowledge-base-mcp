"""
Service layer for business logic.

Provides high-level operations for chunking, search, organization, ingestion,
contextual retrieval, and query transformation.
"""
from .chunking import ChunkingService
from .search import SearchService
from .organization import OrganizationService
from .context import ContextualizationService, get_context_provider
from .hyde import HyDETransformer, get_hyde_provider

__all__ = [
    "ChunkingService",
    "SearchService",
    "OrganizationService",
    "ContextualizationService",
    "get_context_provider",
    "HyDETransformer",
    "get_hyde_provider",
]

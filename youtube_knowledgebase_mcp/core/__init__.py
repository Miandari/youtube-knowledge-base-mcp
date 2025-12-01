"""
Core module for the Media Knowledge Base MCP.

Provides configuration, models, database access, and embedding providers.
"""
from .config import settings, Settings, EmbeddingConfig
from .models import Source, Chunk, SearchResult
from .database import Database, get_db
from .embeddings import (
    EmbeddingProvider,
    VoyageEmbedding,
    OpenAIEmbedding,
    BGEEmbedding,
    OllamaEmbedding,
    get_embedding_provider,
)

__all__ = [
    "settings", "Settings", "EmbeddingConfig",
    "Source", "Chunk", "SearchResult",
    "Database", "get_db",
    "EmbeddingProvider", "VoyageEmbedding", "OpenAIEmbedding",
    "BGEEmbedding", "OllamaEmbedding", "get_embedding_provider",
]

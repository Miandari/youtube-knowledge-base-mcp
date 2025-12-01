"""
Repository layer for data access.

Provides CRUD operations and search methods for sources and chunks.
"""
from .sources import SourceRepository
from .chunks import ChunkRepository

__all__ = [
    "SourceRepository",
    "ChunkRepository",
]

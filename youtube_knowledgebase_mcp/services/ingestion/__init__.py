"""
Ingestion services for different content types.
"""
from .base import IngestionService, IngestionResult
from .youtube import YouTubeIngestionService

__all__ = [
    "IngestionService",
    "IngestionResult",
    "YouTubeIngestionService",
]

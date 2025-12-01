"""
Base classes for content ingestion.

Provides abstract interface for different content type ingestion services.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from ...core.models import Source, Chunk


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    success: bool
    source: Optional[Source] = None
    chunk_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IngestionService(ABC):
    """
    Abstract base class for content ingestion services.

    Each content type (YouTube, articles, podcasts, etc.) should implement
    this interface to provide consistent ingestion behavior.
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """
        Return the source type identifier (e.g., 'youtube', 'article').
        """
        pass

    @abstractmethod
    def extract_metadata(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata from the content URL.

        Args:
            url: The content URL

        Returns:
            Dict containing at minimum: source_id, title
            May also include: channel, description, duration, etc.
        """
        pass

    @abstractmethod
    def extract_content(self, url: str) -> str:
        """
        Extract the main text content from the URL.

        Args:
            url: The content URL

        Returns:
            The extracted text content
        """
        pass

    @abstractmethod
    def extract_segments(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract timestamped segments if available.

        Args:
            url: The content URL

        Returns:
            List of dicts with 'text', 'start', 'end' keys, or None if not applicable
        """
        pass

    @abstractmethod
    def process(
        self,
        url: str,
        tags: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
    ) -> IngestionResult:
        """
        Process content from URL into the knowledge base.

        This is the main entry point that orchestrates:
        1. Metadata extraction
        2. Content extraction
        3. Chunking
        4. Embedding generation
        5. Storage

        Args:
            url: The content URL
            tags: Optional tags to apply to the source
            collections: Optional collections to add the source to

        Returns:
            IngestionResult with success status and details
        """
        pass

    def validate_url(self, url: str) -> bool:
        """
        Validate that the URL is supported by this ingestion service.

        Args:
            url: The URL to validate

        Returns:
            True if the URL is valid for this service
        """
        return True  # Override in subclasses for URL validation

"""
Organization service for managing tags, collections, and summaries.

Provides high-level operations for organizing content in the knowledge base.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..core.models import Source
from ..repositories.sources import SourceRepository
from ..repositories.chunks import ChunkRepository


class OrganizationService:
    """
    Service for organizing content with tags, collections, and summaries.

    Features:
    - Tag management (add, remove, list)
    - Collection management (create, add/remove sources)
    - User summaries
    - Bulk operations
    """

    def __init__(
        self,
        source_repo: Optional[SourceRepository] = None,
        chunk_repo: Optional[ChunkRepository] = None,
    ):
        """
        Initialize the organization service.

        Args:
            source_repo: Repository for source operations
            chunk_repo: Repository for chunk operations
        """
        self._source_repo = source_repo or SourceRepository()
        self._chunk_repo = chunk_repo or ChunkRepository()

    # === Tag Management ===

    def add_tags(self, source_id: str, tags: List[str]) -> Optional[Source]:
        """
        Add tags to a source.

        Args:
            source_id: The source identifier
            tags: Tags to add

        Returns:
            Updated Source or None if not found
        """
        source = self._source_repo.get(source_id)
        if not source:
            return None

        # Add new tags (avoid duplicates)
        existing_tags = set(source.tags)
        new_tags = list(existing_tags | set(tags))

        if new_tags != source.tags:
            source = self._source_repo.update(source_id, tags=new_tags)

            # Update denormalized tags in chunks
            self._update_chunk_tags(source_id, new_tags)

        return source

    def remove_tags(self, source_id: str, tags: List[str]) -> Optional[Source]:
        """
        Remove tags from a source.

        Args:
            source_id: The source identifier
            tags: Tags to remove

        Returns:
            Updated Source or None if not found
        """
        source = self._source_repo.get(source_id)
        if not source:
            return None

        # Remove specified tags
        new_tags = [t for t in source.tags if t not in tags]

        if new_tags != source.tags:
            source = self._source_repo.update(source_id, tags=new_tags)

            # Update denormalized tags in chunks
            self._update_chunk_tags(source_id, new_tags)

        return source

    def set_tags(self, source_id: str, tags: List[str]) -> Optional[Source]:
        """
        Replace all tags on a source.

        Args:
            source_id: The source identifier
            tags: New tags (replaces existing)

        Returns:
            Updated Source or None if not found
        """
        source = self._source_repo.update(source_id, tags=tags)

        if source:
            # Update denormalized tags in chunks
            self._update_chunk_tags(source_id, tags)

        return source

    def _update_chunk_tags(self, source_id: str, tags: List[str]):
        """Update denormalized tags in all chunks for a source."""
        chunks = self._chunk_repo.get_by_source(source_id)
        if not chunks:
            return

        # Delete and re-add chunks with updated tags
        self._chunk_repo.delete_by_source(source_id)

        for chunk in chunks:
            chunk.tags = tags

        self._chunk_repo.add(chunks)

    def list_all_tags(self) -> List[str]:
        """
        Get all unique tags across all sources.

        Returns:
            Sorted list of unique tags
        """
        return self._source_repo.list_all_tags()

    # === Collection Management ===

    def add_to_collection(
        self,
        source_id: str,
        collection: str,
    ) -> Optional[Source]:
        """
        Add a source to a collection.

        Args:
            source_id: The source identifier
            collection: Collection name

        Returns:
            Updated Source or None if not found
        """
        source = self._source_repo.get(source_id)
        if not source:
            return None

        # Add to collection (avoid duplicates)
        if collection not in source.collections:
            new_collections = source.collections + [collection]
            source = self._source_repo.update(source_id, collections=new_collections)

            # Update denormalized collections in chunks
            self._update_chunk_collections(source_id, new_collections)

        return source

    def remove_from_collection(
        self,
        source_id: str,
        collection: str,
    ) -> Optional[Source]:
        """
        Remove a source from a collection.

        Args:
            source_id: The source identifier
            collection: Collection name

        Returns:
            Updated Source or None if not found
        """
        source = self._source_repo.get(source_id)
        if not source:
            return None

        if collection in source.collections:
            new_collections = [c for c in source.collections if c != collection]
            source = self._source_repo.update(source_id, collections=new_collections)

            # Update denormalized collections in chunks
            self._update_chunk_collections(source_id, new_collections)

        return source

    def _update_chunk_collections(self, source_id: str, collections: List[str]):
        """Update denormalized collections in all chunks for a source."""
        chunks = self._chunk_repo.get_by_source(source_id)
        if not chunks:
            return

        # Delete and re-add chunks with updated collections
        self._chunk_repo.delete_by_source(source_id)

        for chunk in chunks:
            chunk.collections = collections

        self._chunk_repo.add(chunks)

    def list_all_collections(self) -> List[str]:
        """
        Get all unique collections across all sources.

        Returns:
            Sorted list of unique collections
        """
        return self._source_repo.list_all_collections()

    def get_collection_sources(self, collection: str) -> List[Source]:
        """
        Get all sources in a collection.

        Args:
            collection: Collection name

        Returns:
            List of Sources in the collection
        """
        return self._source_repo.list(collections=[collection])

    # === Summary Management ===

    def set_summary(
        self,
        source_id: str,
        summary: str,
    ) -> Optional[Source]:
        """
        Set or update the user summary for a source.

        Args:
            source_id: The source identifier
            summary: The summary text

        Returns:
            Updated Source or None if not found
        """
        return self._source_repo.update(source_id, user_summary=summary)

    def get_summary(self, source_id: str) -> Optional[str]:
        """
        Get the user summary for a source.

        Args:
            source_id: The source identifier

        Returns:
            Summary text or None if not found/not set
        """
        source = self._source_repo.get(source_id)
        if source:
            return source.user_summary
        return None

    def clear_summary(self, source_id: str) -> Optional[Source]:
        """
        Clear the user summary for a source.

        Args:
            source_id: The source identifier

        Returns:
            Updated Source or None if not found
        """
        return self._source_repo.update(source_id, user_summary=None)

    # === Bulk Operations ===

    def bulk_add_tags(
        self,
        source_ids: List[str],
        tags: List[str],
    ) -> int:
        """
        Add tags to multiple sources.

        Args:
            source_ids: List of source identifiers
            tags: Tags to add

        Returns:
            Number of sources updated
        """
        updated = 0
        for source_id in source_ids:
            if self.add_tags(source_id, tags):
                updated += 1
        return updated

    def bulk_add_to_collection(
        self,
        source_ids: List[str],
        collection: str,
    ) -> int:
        """
        Add multiple sources to a collection.

        Args:
            source_ids: List of source identifiers
            collection: Collection name

        Returns:
            Number of sources updated
        """
        updated = 0
        for source_id in source_ids:
            if self.add_to_collection(source_id, collection):
                updated += 1
        return updated

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """
        Get organization statistics.

        Returns:
            Dict with counts of sources, chunks, tags, collections
        """
        return {
            "total_sources": self._source_repo.count(),
            "total_chunks": self._chunk_repo.count(),
            "sources_by_type": self._source_repo.count_by_type(),
            "unique_tags": len(self.list_all_tags()),
            "unique_collections": len(self.list_all_collections()),
            "tags": self.list_all_tags(),
            "collections": self.list_all_collections(),
        }

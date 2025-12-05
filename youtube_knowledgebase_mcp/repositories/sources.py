"""
Repository for Source entities.

Provides CRUD operations for the sources table.
"""
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..core.database import get_db
from ..core.models import Source


class SourceRepository:
    """Repository for managing sources in LanceDB."""

    def __init__(self):
        self._db = get_db()

    @property
    def _table(self):
        """Get the sources table."""
        return self._db.sources

    def _source_to_record(self, source: Source) -> Dict[str, Any]:
        """Convert a Source model to a LanceDB record."""
        return {
            "id": source.id,
            "source_type": source.source_type,
            "source_id": source.source_id,
            "title": source.title,
            "channel": source.channel,
            "description": source.description,
            "url": source.url,
            "metadata": json.dumps(source.metadata),  # JSON string for storage
            "tags": source.tags,
            "collections": source.collections,
            "user_summary": source.user_summary,
            "is_processed": source.is_processed,
            "chunk_count": source.chunk_count,
            "embedding_model": source.embedding_model,
            "created_at": source.created_at,  # datetime object for PyArrow
            "updated_at": source.updated_at,
        }

    def _record_to_source(self, record: Dict[str, Any]) -> Source:
        """Convert a LanceDB record to a Source model."""
        metadata = record.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Source(
            id=record["id"],
            source_type=record["source_type"],
            source_id=record["source_id"],
            title=record["title"],
            channel=record.get("channel"),
            description=record.get("description"),
            url=record.get("url"),
            metadata=metadata,
            tags=record.get("tags", []),
            collections=record.get("collections", []),
            user_summary=record.get("user_summary"),
            is_processed=record.get("is_processed", False),
            chunk_count=record.get("chunk_count", 0),
            embedding_model=record.get("embedding_model"),
            created_at=record.get("created_at", datetime.utcnow()),
            updated_at=record.get("updated_at", datetime.utcnow()),
        )

    def add(self, source: Source) -> Source:
        """
        Add a new source to the database.

        Args:
            source: The Source to add

        Returns:
            The added Source with any generated fields
        """
        record = self._source_to_record(source)
        self._table.add([record])
        return source

    def get(self, source_id: str) -> Optional[Source]:
        """
        Get a source by its source_id.

        Args:
            source_id: The source identifier (e.g., YouTube video ID)

        Returns:
            The Source if found, None otherwise
        """
        results = self._table.search().where(
            f"source_id = '{source_id}'"
        ).limit(1).to_list()

        if not results:
            return None

        return self._record_to_source(results[0])

    def get_by_id(self, id: str) -> Optional[Source]:
        """
        Get a source by its internal UUID.

        Args:
            id: The internal UUID

        Returns:
            The Source if found, None otherwise
        """
        results = self._table.search().where(
            f"id = '{id}'"
        ).limit(1).to_list()

        if not results:
            return None

        return self._record_to_source(results[0])

    def exists(self, source_id: str) -> bool:
        """
        Check if a source exists.

        Args:
            source_id: The source identifier

        Returns:
            True if the source exists
        """
        results = self._table.search().where(
            f"source_id = '{source_id}'"
        ).limit(1).to_list()
        return len(results) > 0

    def update(self, source_id: str, **fields) -> Optional[Source]:
        """
        Update a source's fields.

        Args:
            source_id: The source identifier
            **fields: Fields to update

        Returns:
            The updated Source if found, None otherwise
        """
        existing = self.get(source_id)
        if not existing:
            return None

        # Update fields
        fields["updated_at"] = datetime.utcnow()

        # Handle metadata specially - merge if provided
        if "metadata" in fields and isinstance(fields["metadata"], dict):
            fields["metadata"] = json.dumps(fields["metadata"])

        # Build update dict
        update_data = {}
        for key, value in fields.items():
            if hasattr(existing, key):
                update_data[key] = value

        if update_data:
            # LanceDB update: delete and re-add
            self.delete(source_id)

            # Apply updates to existing source
            existing_dict = existing.model_dump()
            existing_dict.update(update_data)

            # Handle metadata back to dict for Source model
            if "metadata" in existing_dict and isinstance(existing_dict["metadata"], str):
                existing_dict["metadata"] = json.loads(existing_dict["metadata"])

            updated_source = Source(**existing_dict)
            return self.add(updated_source)

        return existing

    def delete(self, source_id: str) -> bool:
        """
        Delete a source by its source_id.

        Args:
            source_id: The source identifier

        Returns:
            True if deleted, False if not found
        """
        if not self.exists(source_id):
            return False

        self._table.delete(f"source_id = '{source_id}'")
        return True

    def list(
        self,
        source_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        is_processed: Optional[bool] = None,
        channel: Optional[str] = None,
        title_contains: Optional[str] = None,
        has_summary: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
    ) -> List[Source]:
        """
        List sources with optional filtering.

        Args:
            source_type: Filter by source type (e.g., "youtube")
            tags: Filter by tags (sources must have at least one matching tag)
            collections: Filter by collections
            is_processed: Filter by processing status
            channel: Filter by channel name (exact match)
            title_contains: Filter by title (case-insensitive contains)
            has_summary: Filter by whether source has a user summary
            limit: Maximum number of results
            offset: Number of results to skip
            sort_by: Sort field - "created_at", "updated_at", or "title"

        Returns:
            List of matching Sources
        """
        # Build filter conditions
        conditions = []

        if source_type:
            conditions.append(f"source_type = '{source_type}'")

        if is_processed is not None:
            conditions.append(f"is_processed = {str(is_processed).lower()}")

        if channel:
            # Escape single quotes in channel name
            escaped_channel = channel.replace("'", "''")
            conditions.append(f"channel = '{escaped_channel}'")

        # Execute query
        query = self._table.search()

        if conditions:
            where_clause = " AND ".join(conditions)
            query = query.where(where_clause)

        # Get more results to filter in Python
        results = query.limit(limit * 3 + offset).to_list()

        # Convert to Source models
        sources = [self._record_to_source(r) for r in results]

        # Filter by tags/collections in Python (array contains is complex in SQL)
        if tags:
            sources = [s for s in sources if any(t in s.tags for t in tags)]

        if collections:
            sources = [s for s in sources if any(c in s.collections for c in collections)]

        # Filter by title_contains (case-insensitive)
        if title_contains:
            title_lower = title_contains.lower()
            sources = [s for s in sources if title_lower in s.title.lower()]

        # Filter by has_summary
        if has_summary is not None:
            if has_summary:
                sources = [s for s in sources if s.user_summary]
            else:
                sources = [s for s in sources if not s.user_summary]

        # Sort results
        if sort_by == "title":
            sources.sort(key=lambda s: s.title.lower())
        elif sort_by == "updated_at":
            sources.sort(key=lambda s: s.updated_at, reverse=True)
        else:  # default to created_at
            sources.sort(key=lambda s: s.created_at, reverse=True)

        # Apply offset and limit
        return sources[offset:offset + limit]

    def count(self) -> int:
        """
        Get the total number of sources.

        Returns:
            The count of sources
        """
        return self._table.count_rows()

    def list_all_tags(self) -> List[str]:
        """
        Get all unique tags across all sources.

        Returns:
            List of unique tags
        """
        # Use column projection to fetch only tags (avoids loading full source data)
        try:
            results = self._table.search().select(["tags"]).to_list()
        except Exception:
            # Fallback if select not supported
            results = self._table.search().to_list()

        all_tags = set()
        for r in results:
            tags = r.get("tags", [])
            if tags:
                all_tags.update(tags)
        return sorted(list(all_tags))

    def list_all_collections(self) -> List[str]:
        """
        Get all unique collections across all sources.

        Returns:
            List of unique collections
        """
        # Use column projection to fetch only collections
        try:
            results = self._table.search().select(["collections"]).to_list()
        except Exception:
            results = self._table.search().to_list()

        all_collections = set()
        for r in results:
            collections = r.get("collections", [])
            if collections:
                all_collections.update(collections)
        return sorted(list(all_collections))

    def count_by_type(self) -> Dict[str, int]:
        """
        Count sources grouped by source_type.

        Returns:
            Dict mapping source_type to count
        """
        # Use column projection to fetch only source_type
        try:
            results = self._table.search().select(["source_type"]).to_list()
        except Exception:
            results = self._table.search().to_list()

        counts: Dict[str, int] = {}
        for r in results:
            source_type = r.get("source_type", "unknown")
            counts[source_type] = counts.get(source_type, 0) + 1
        return counts

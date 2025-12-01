"""
Repository for Chunk entities.

Provides CRUD operations and hybrid search for the chunks table.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..core.database import get_db
from ..core.models import Chunk, SearchResult


class ChunkRepository:
    """Repository for managing chunks in LanceDB with hybrid search support."""

    def __init__(self):
        self._db = get_db()

    @property
    def _table(self):
        """Get the chunks table."""
        return self._db.chunks

    def _chunk_to_record(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert a Chunk model to a LanceDB record."""
        return {
            "id": chunk.id,
            "source_id": chunk.source_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "vector": chunk.vector,
            "timestamp_start": chunk.timestamp_start,
            "timestamp_end": chunk.timestamp_end,
            "source_type": chunk.source_type,
            "source_title": chunk.source_title,
            "source_channel": chunk.source_channel,
            "tags": chunk.tags,
            "collections": chunk.collections,
            "embedding_model": chunk.embedding_model,
            "created_at": chunk.created_at,  # datetime object for PyArrow
        }

    def _record_to_chunk(self, record: Dict[str, Any]) -> Chunk:
        """Convert a LanceDB record to a Chunk model."""
        return Chunk(
            id=record["id"],
            source_id=record["source_id"],
            content=record["content"],
            chunk_index=record.get("chunk_index", 0),
            vector=record.get("vector"),
            timestamp_start=record.get("timestamp_start"),
            timestamp_end=record.get("timestamp_end"),
            source_type=record.get("source_type", "youtube"),
            source_title=record.get("source_title", ""),
            source_channel=record.get("source_channel"),
            tags=record.get("tags", []),
            collections=record.get("collections", []),
            embedding_model=record.get("embedding_model", ""),
            created_at=record.get("created_at", datetime.utcnow()),
        )

    def add(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Add chunks to the database (batch operation).

        Args:
            chunks: List of Chunks to add (must have vectors populated)

        Returns:
            The added Chunks
        """
        if not chunks:
            return []

        records = [self._chunk_to_record(c) for c in chunks]
        self._table.add(records)

        # Recreate FTS index after adding new content
        try:
            self._table.create_fts_index("content", replace=True)
        except Exception:
            pass  # Index creation may fail on empty table

        return chunks

    def get(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get a chunk by its ID.

        Args:
            chunk_id: The chunk UUID

        Returns:
            The Chunk if found, None otherwise
        """
        results = self._table.search().where(
            f"id = '{chunk_id}'"
        ).limit(1).to_list()

        if not results:
            return None

        return self._record_to_chunk(results[0])

    def get_by_source(self, source_id: str) -> List[Chunk]:
        """
        Get all chunks for a source, ordered by chunk_index.

        Args:
            source_id: The source identifier

        Returns:
            List of Chunks for the source
        """
        results = self._table.search().where(
            f"source_id = '{source_id}'"
        ).limit(10000).to_list()

        chunks = [self._record_to_chunk(r) for r in results]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def delete_by_source(self, source_id: str) -> int:
        """
        Delete all chunks for a source.

        Args:
            source_id: The source identifier

        Returns:
            Number of chunks deleted
        """
        # Count before deletion
        existing = self.get_by_source(source_id)
        count = len(existing)

        if count > 0:
            self._table.delete(f"source_id = '{source_id}'")

        return count

    def search(
        self,
        query_vector: List[float],
        query_text: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        source_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        limit: int = 10,
        hybrid_weight: float = 0.7,
    ) -> List[SearchResult]:
        """
        Perform hybrid search (vector + full-text) on chunks.

        Args:
            query_vector: The query embedding vector
            query_text: Optional text for full-text search component
            source_ids: Optional list of source_ids to filter by
            source_type: Optional source type filter
            tags: Optional tags filter
            collections: Optional collections filter
            limit: Maximum number of results
            hybrid_weight: Weight for vector vs FTS (0.0 = FTS only, 1.0 = vector only)

        Returns:
            List of SearchResults with scores
        """
        # Build filter conditions
        conditions = []

        if source_type:
            conditions.append(f"source_type = '{source_type}'")

        if source_ids:
            source_list = ", ".join(f"'{sid}'" for sid in source_ids)
            conditions.append(f"source_id IN ({source_list})")

        where_clause = " AND ".join(conditions) if conditions else None

        # Perform search
        search_query = self._table.search(
            query_vector,
            query_type="hybrid" if query_text else "vector",
        )

        if query_text:
            search_query = search_query.text(query_text)

        if where_clause:
            search_query = search_query.where(where_clause)

        results = search_query.limit(limit * 2).to_list()  # Get extra for filtering

        # Convert to SearchResults
        search_results = []
        for r in results:
            chunk = self._record_to_chunk(r)

            # Apply tag/collection filters in Python
            if tags and not any(t in chunk.tags for t in tags):
                continue
            if collections and not any(c in chunk.collections for c in collections):
                continue

            # Get distance score (lower is better in LanceDB)
            distance = r.get("_distance", 0.0)
            # Convert distance to similarity score (1 / (1 + distance))
            score = 1.0 / (1.0 + distance)

            search_result = SearchResult(
                chunk=chunk,
                score=score,
                recency_weight=1.0,  # Will be adjusted by search service
                final_score=score,
                source_title=chunk.source_title,
                source_url=None,  # Will be populated by service layer
                timestamp_link=None,
            )
            search_results.append(search_result)

            if len(search_results) >= limit:
                break

        return search_results

    def vector_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        where: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform pure vector search on chunks.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results
            where: Optional SQL-like where clause

        Returns:
            List of SearchResults with scores
        """
        search_query = self._table.search(query_vector, query_type="vector")

        if where:
            search_query = search_query.where(where)

        results = search_query.limit(limit).to_list()

        search_results = []
        for r in results:
            chunk = self._record_to_chunk(r)
            distance = r.get("_distance", 0.0)
            score = 1.0 / (1.0 + distance)

            search_result = SearchResult(
                chunk=chunk,
                score=score,
                recency_weight=1.0,
                final_score=score,
                source_title=chunk.source_title,
                source_url=None,
                timestamp_link=None,
            )
            search_results.append(search_result)

        return search_results

    def fts_search(
        self,
        query_text: str,
        limit: int = 10,
        where: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform pure full-text search on chunks.

        Args:
            query_text: The search text
            limit: Maximum number of results
            where: Optional SQL-like where clause

        Returns:
            List of SearchResults with scores
        """
        search_query = self._table.search(query_text, query_type="fts")

        if where:
            search_query = search_query.where(where)

        results = search_query.limit(limit).to_list()

        search_results = []
        for r in results:
            chunk = self._record_to_chunk(r)
            # FTS returns _score (higher is better)
            score = r.get("_score", r.get("_distance", 0.0))

            search_result = SearchResult(
                chunk=chunk,
                score=score,
                recency_weight=1.0,
                final_score=score,
                source_title=chunk.source_title,
                source_url=None,
                timestamp_link=None,
            )
            search_results.append(search_result)

        return search_results

    def count(self) -> int:
        """
        Get the total number of chunks.

        Returns:
            The count of chunks
        """
        return self._table.count_rows()

    def count_by_source(self, source_id: str) -> int:
        """
        Get the number of chunks for a source.

        Args:
            source_id: The source identifier

        Returns:
            The count of chunks for the source
        """
        results = self._table.search().where(
            f"source_id = '{source_id}'"
        ).limit(10000).to_list()
        return len(results)

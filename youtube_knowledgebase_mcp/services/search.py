"""
Search service for hybrid search with recency weighting.

Orchestrates vector and full-text search, applies recency boosting,
and handles result ranking and deduplication.
"""
import asyncio
import math
from datetime import datetime, timedelta
from typing import List, Optional

from ..core.config import settings
from ..core.models import SearchResult
from ..core.embeddings import get_embedding_provider, EmbeddingProvider
from ..repositories.chunks import ChunkRepository
from ..repositories.sources import SourceRepository


class SearchService:
    """
    Service for searching the knowledge base.

    Features:
    - Hybrid search (vector + full-text)
    - Recency weighting (recent content boosted)
    - Result deduplication
    - Source metadata enrichment
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        chunk_repo: Optional[ChunkRepository] = None,
        source_repo: Optional[SourceRepository] = None,
    ):
        """
        Initialize the search service.

        Args:
            embedding_provider: Provider for generating query embeddings
            chunk_repo: Repository for chunk operations
            source_repo: Repository for source operations
        """
        self._embedding_provider = embedding_provider
        self._chunk_repo = chunk_repo or ChunkRepository()
        self._source_repo = source_repo or SourceRepository()

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Lazy initialization of embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider(
                provider=settings.embedding.provider,
                model=settings.embedding.get_model_name(),
                dimensions=settings.embedding.dimensions,
            )
        return self._embedding_provider

    def search(
        self,
        query: str,
        limit: int = None,
        source_ids: Optional[List[str]] = None,
        source_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        recency_boost: Optional[bool] = None,
        hybrid: bool = True,
    ) -> List[SearchResult]:
        """
        Search the knowledge base with hybrid search and optional recency weighting.

        Args:
            query: The search query text
            limit: Maximum number of results (default from settings)
            source_ids: Optional list of source_ids to filter by
            source_type: Optional source type filter
            tags: Optional tags filter
            collections: Optional collections filter
            recency_boost: Whether to apply recency weighting (default from settings)
            hybrid: Whether to use hybrid search (vector + FTS) or just vector

        Returns:
            List of SearchResults sorted by final_score
        """
        limit = limit or settings.default_search_limit
        recency_boost = recency_boost if recency_boost is not None else settings.recency_boost_enabled

        # Generate query embedding
        query_vector = self.embedding_provider.embed_query(query)

        # Perform search
        if hybrid:
            results = self._chunk_repo.search(
                query_vector=query_vector,
                query_text=query,
                source_ids=source_ids,
                source_type=source_type,
                tags=tags,
                collections=collections,
                limit=limit * 2,  # Get extra for deduplication
            )
        else:
            # Build where clause for vector-only search
            conditions = []
            if source_type:
                conditions.append(f"source_type = '{source_type}'")
            if source_ids:
                source_list = ", ".join(f"'{sid}'" for sid in source_ids)
                conditions.append(f"source_id IN ({source_list})")

            where_clause = " AND ".join(conditions) if conditions else None

            results = self._chunk_repo.vector_search(
                query_vector=query_vector,
                limit=limit * 2,
                where=where_clause,
            )

            # Apply tag/collection filters in Python for vector-only search
            if tags:
                results = [r for r in results if any(t in r.chunk.tags for t in tags)]
            if collections:
                results = [r for r in results if any(c in r.chunk.collections for c in collections)]

        # Apply recency weighting
        if recency_boost:
            results = self._apply_recency_weighting(results)

        # Enrich with source metadata
        results = self._enrich_results(results)

        # Deduplicate by content similarity
        results = self._deduplicate_results(results)

        # Sort by final score and limit
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:limit]

    def _apply_recency_weighting(
        self,
        results: List[SearchResult],
        max_age_days: int = 365,
    ) -> List[SearchResult]:
        """
        Apply recency weighting to search results.

        Recent content gets boosted, older content is dampened (with a floor).

        Args:
            results: Search results to weight
            max_age_days: Age at which content reaches minimum weight

        Returns:
            Results with recency_weight and final_score updated
        """
        now = datetime.utcnow()

        for result in results:
            # Get source creation date from chunk
            created_at = result.chunk.created_at

            # Calculate age in days
            age = now - created_at
            age_days = age.days

            # Calculate recency weight using exponential decay
            # Weight goes from 1.0 (new) to recency_floor (old)
            if age_days <= 0:
                weight = 1.0
            elif age_days >= max_age_days:
                weight = settings.recency_floor
            else:
                # Exponential decay from 1.0 to recency_floor
                decay_rate = -math.log(settings.recency_floor) / max_age_days
                weight = math.exp(-decay_rate * age_days)
                weight = max(weight, settings.recency_floor)

            result.recency_weight = weight
            result.final_score = result.score * weight

        return results

    def _enrich_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Enrich search results with source metadata.

        Args:
            results: Search results to enrich

        Returns:
            Results with source_url and timestamp_link populated
        """
        # Cache source lookups
        source_cache = {}

        for result in results:
            source_id = result.chunk.source_id

            # Get source from cache or fetch
            if source_id not in source_cache:
                source = self._source_repo.get(source_id)
                source_cache[source_id] = source

            source = source_cache.get(source_id)

            if source:
                result.source_url = source.url

                # Generate timestamp link for YouTube
                if source.source_type == "youtube" and result.chunk.timestamp_start is not None:
                    video_id = source.source_id
                    timestamp = int(result.chunk.timestamp_start)
                    result.timestamp_link = f"https://youtube.com/watch?v={video_id}&t={timestamp}s"

        return results

    def _deduplicate_results(
        self,
        results: List[SearchResult],
        similarity_threshold: float = 0.9,
    ) -> List[SearchResult]:
        """
        Remove near-duplicate results based on content similarity.

        Args:
            results: Search results to deduplicate
            similarity_threshold: Jaccard similarity threshold for duplicates

        Returns:
            Deduplicated results
        """
        if not results:
            return results

        unique_results = []
        seen_content = []

        for result in results:
            content = result.chunk.content.lower()
            content_words = set(content.split())

            # Check similarity with seen content
            is_duplicate = False
            for seen_words in seen_content:
                if not content_words or not seen_words:
                    continue

                # Jaccard similarity
                intersection = len(content_words & seen_words)
                union = len(content_words | seen_words)
                similarity = intersection / union if union > 0 else 0

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_content.append(content_words)

        return unique_results

    def search_by_source(
        self,
        query: str,
        source_id: str,
        limit: int = 5,
    ) -> List[SearchResult]:
        """
        Search within a specific source.

        Args:
            query: The search query text
            source_id: The source to search within
            limit: Maximum number of results

        Returns:
            List of SearchResults from the specified source
        """
        return self.search(
            query=query,
            source_ids=[source_id],
            limit=limit,
            recency_boost=False,  # Don't weight recency within single source
        )

    def similar_chunks(
        self,
        chunk_id: str,
        limit: int = 5,
        exclude_same_source: bool = True,
    ) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: The chunk to find similar content for
            limit: Maximum number of results
            exclude_same_source: Whether to exclude chunks from the same source

        Returns:
            List of similar SearchResults
        """
        # Get the reference chunk
        chunk = self._chunk_repo.get(chunk_id)
        if not chunk or not chunk.vector:
            return []

        # Search using the chunk's vector
        results = self._chunk_repo.vector_search(
            query_vector=chunk.vector,
            limit=limit + 10,  # Get extra for filtering
        )

        # Filter out the reference chunk and optionally same-source chunks
        filtered = []
        for result in results:
            if result.chunk.id == chunk_id:
                continue
            if exclude_same_source and result.chunk.source_id == chunk.source_id:
                continue
            filtered.append(result)

        # Enrich and return
        filtered = self._enrich_results(filtered)
        return filtered[:limit]

    async def search_async(
        self,
        query: str,
        limit: int = None,
        source_ids: Optional[List[str]] = None,
        source_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        recency_boost: Optional[bool] = None,
        hybrid: bool = True,
    ) -> List[SearchResult]:
        """
        Async version of search - embedding API call is the bottleneck.

        Args:
            query: The search query text
            limit: Maximum number of results (default from settings)
            source_ids: Optional list of source_ids to filter by
            source_type: Optional source type filter
            tags: Optional tags filter
            collections: Optional collections filter
            recency_boost: Whether to apply recency weighting (default from settings)
            hybrid: Whether to use hybrid search (vector + FTS) or just vector

        Returns:
            List of SearchResults sorted by final_score
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(
                query=query,
                limit=limit,
                source_ids=source_ids,
                source_type=source_type,
                tags=tags,
                collections=collections,
                recency_boost=recency_boost,
                hybrid=hybrid,
            )
        )

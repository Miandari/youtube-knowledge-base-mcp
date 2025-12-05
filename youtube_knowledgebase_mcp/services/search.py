"""
Search service with two-stage retrieval and cross-encoder reranking.

Pipeline:
1. Retrieve: Bi-encoder fetches high-recall candidates (5x limit)
2. Rerank: Cross-encoder scores query-document pairs for precision
3. Deduplicate: Remove near-duplicates (reranker scores relevance, not diversity)
"""
import asyncio
import logging
from typing import List, Optional

from ..core.config import settings
from ..core.models import SearchResult
from ..core.embeddings import get_embedding_provider, EmbeddingProvider
from ..repositories.chunks import ChunkRepository
from ..repositories.sources import SourceRepository

logger = logging.getLogger(__name__)

# Lazy imports to avoid startup cost if features disabled
Ranker = None
RerankRequest = None
HyDETransformer = None
get_hyde_provider = None


class SearchService:
    """
    Service for searching the knowledge base.

    Features:
    - Two-stage retrieval (bi-encoder + cross-encoder reranking)
    - Hybrid search (vector + full-text via LanceDB RRF)
    - Result deduplication for diversity
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

    @property
    def reranker(self):
        """Lazy initialization of cross-encoder reranker."""
        if not hasattr(self, '_reranker'):
            self._reranker = None
            if settings.rerank.enabled:
                try:
                    global Ranker, RerankRequest
                    if Ranker is None:
                        from flashrank import Ranker, RerankRequest
                    self._reranker = Ranker(
                        model_name=settings.rerank.model_name,
                        max_length=settings.rerank.max_length,
                        cache_dir=settings.rerank.cache_dir,
                    )
                    logger.info(f"Initialized reranker: {settings.rerank.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize reranker: {e}. Using vector scores only.")
        return self._reranker

    @property
    def hyde_transformer(self):
        """Lazy initialization of HyDE transformer."""
        if not hasattr(self, '_hyde_transformer'):
            self._hyde_transformer = None
            if settings.hyde.enabled:
                try:
                    global HyDETransformer, get_hyde_provider
                    if HyDETransformer is None:
                        from .hyde import HyDETransformer, get_hyde_provider
                    provider = get_hyde_provider(
                        provider=settings.hyde.provider,
                        model=settings.hyde.get_model_name(),
                        temperature=settings.hyde.temperature,
                    )
                    self._hyde_transformer = HyDETransformer(
                        provider=provider,
                        max_tokens=settings.hyde.max_tokens,
                    )
                    logger.info(f"Initialized HyDE transformer: {settings.hyde.provider}")
                except Exception as e:
                    logger.warning(f"Failed to initialize HyDE: {e}")
        return self._hyde_transformer

    def search(
        self,
        query: str,
        limit: int = None,
        source_ids: Optional[List[str]] = None,
        source_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        hybrid: bool = True,
    ) -> List[SearchResult]:
        """
        Three-stage search: retrieve candidates, rerank, then deduplicate.

        Stage 1: Bi-encoder retrieves high-recall candidates (limit * multiplier)
        Stage 2: Cross-encoder reranks for precision
        Stage 3: Deduplicate for diversity (reranker scores relevance, not diversity)

        Args:
            query: The search query text
            limit: Maximum number of results (default from settings)
            source_ids: Optional list of source_ids to filter by
            source_type: Optional source type filter
            tags: Optional tags filter
            collections: Optional collections filter
            hybrid: Whether to use hybrid search (vector + FTS) or just vector

        Returns:
            List of SearchResults sorted by final_score
        """
        limit = limit or settings.default_search_limit
        candidate_limit = limit * settings.rerank.candidate_multiplier

        # Transform query with HyDE if enabled (for embedding only)
        embedding_query = query
        if self.hyde_transformer:
            embedding_query = self.hyde_transformer.transform(query)

        # Generate query embedding (using transformed or original query)
        query_vector = self.embedding_provider.embed_query(embedding_query)

        # === STAGE 1: Retrieve candidates ===
        if hybrid:
            results = self._chunk_repo.search(
                query_vector=query_vector,
                query_text=query,
                source_ids=source_ids,
                source_type=source_type,
                tags=tags,
                collections=collections,
                limit=candidate_limit,
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
                limit=candidate_limit,
                where=where_clause,
            )

            # Apply tag filter in Python for vector-only search
            # NOTE: collections filtering removed - collections not stored on chunks
            if tags:
                results = [r for r in results if any(t in r.chunk.tags for t in tags)]

        # === STAGE 2: Rerank with cross-encoder ===
        if self.reranker and results:
            results = self._rerank_results(query, results)
        else:
            # Fallback: use vector scores as final scores
            for result in results:
                result.final_score = result.score

        # === STAGE 3: Deduplicate (AFTER reranking) ===
        # CRITICAL: Reranker scores for RELEVANCE, not DIVERSITY.
        # Without this, 5 near-duplicate relevant chunks would all appear at top.
        results = self._deduplicate_results(results)

        # Enrich with source metadata
        results = self._enrich_results(results)

        # Sort by final score and limit
        results.sort(key=lambda r: r.final_score, reverse=True)
        results = results[:limit]

        # Strip vectors before returning (they're huge and not needed in results)
        for result in results:
            result.chunk.vector = None

        return results

    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder.

        Key: If chunk has context field (from Contextual Retrieval),
        prepend it for richer reranking.

        Args:
            query: The search query
            results: Search results to rerank

        Returns:
            Results with updated final_score from cross-encoder
        """
        global RerankRequest
        if RerankRequest is None:
            from flashrank import RerankRequest

        # Build passages for reranker
        passages = []
        for result in results:
            chunk = result.chunk

            # CRUCIAL: Use context if available (from Contextual Retrieval)
            if chunk.context:
                text = f"{chunk.context}\n\n{chunk.content}"
            else:
                text = chunk.content

            passages.append({
                "id": chunk.id,
                "text": text,
                "meta": {"original_score": result.score},
            })

        # Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = self.reranker.rerank(rerank_request)

        # Map back to SearchResult objects
        id_to_result = {r.chunk.id: r for r in results}
        reranked_results = []

        for item in reranked:
            result = id_to_result.get(item["id"])
            if result:
                # Update scores: cross-encoder score is the new final_score
                result.score = item["meta"]["original_score"]  # Keep original
                result.final_score = item["score"]  # Cross-encoder probability (0-1)
                result.recency_weight = 1.0  # No longer used
                reranked_results.append(result)

        return reranked_results

    def _enrich_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Enrich search results with source metadata.

        Args:
            results: Search results to enrich

        Returns:
            Results with source_title, source_url and timestamp_link populated
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
                # Populate source metadata (source_title no longer stored on chunk)
                result.source_title = source.title
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
                hybrid=hybrid,
            )
        )

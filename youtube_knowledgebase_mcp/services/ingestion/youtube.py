"""
YouTube ingestion service.

Processes YouTube videos into the knowledge base using existing transcript extraction.
"""
import asyncio
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs

from .base import IngestionService, IngestionResult
from ...core.config import settings
from ...core.models import Source, Chunk
from ...core.embeddings import get_embedding_provider, EmbeddingProvider
from ...repositories.sources import SourceRepository
from ...repositories.chunks import ChunkRepository
from ..chunking import ChunkingService

# Import existing YouTube functions
from ...youtube_transcript import (
    extract_youtube_transcript,
    process_webvtt_transcript,
    get_youtube_metadata,
)


class YouTubeIngestionService(IngestionService):
    """
    Service for ingesting YouTube videos into the knowledge base.

    Uses the existing youtube_transcript module for extraction,
    then processes and stores content using the new architecture.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        source_repo: Optional[SourceRepository] = None,
        chunk_repo: Optional[ChunkRepository] = None,
        chunking_service: Optional[ChunkingService] = None,
    ):
        """
        Initialize the YouTube ingestion service.

        Args:
            embedding_provider: Provider for generating embeddings
            source_repo: Repository for source operations
            chunk_repo: Repository for chunk operations
            chunking_service: Service for text chunking
        """
        self._embedding_provider = embedding_provider
        self._source_repo = source_repo or SourceRepository()
        self._chunk_repo = chunk_repo or ChunkRepository()
        self._chunking_service = chunking_service or ChunkingService()

    @property
    def source_type(self) -> str:
        return "youtube"

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

    def validate_url(self, url: str) -> bool:
        """
        Validate that the URL is a YouTube URL.

        Args:
            url: The URL to validate

        Returns:
            True if the URL is a valid YouTube URL
        """
        try:
            parsed = urlparse(url)
            if parsed.netloc == 'youtu.be':
                return bool(parsed.path and len(parsed.path) > 1)
            elif 'youtube.com' in parsed.netloc:
                query_params = parse_qs(parsed.query)
                return 'v' in query_params and bool(query_params['v'][0])
            return False
        except Exception:
            return False

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        try:
            parsed = urlparse(url)
            if parsed.netloc == 'youtu.be':
                return parsed.path[1:]
            elif 'youtube.com' in parsed.netloc:
                query_params = parse_qs(parsed.query)
                return query_params.get('v', [''])[0]
        except Exception:
            pass
        return None

    def extract_metadata(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata from a YouTube video.

        Args:
            url: The YouTube URL

        Returns:
            Dict containing video metadata
        """
        metadata = get_youtube_metadata(url)

        # Ensure we have the video_id even if extraction fails
        if not metadata.get('video_id'):
            metadata['video_id'] = self._extract_video_id(url) or ''

        return metadata

    def extract_content(self, url: str) -> str:
        """
        Extract transcript text from a YouTube video.

        Args:
            url: The YouTube URL

        Returns:
            The transcript text
        """
        video_id, webvtt_content = extract_youtube_transcript(url)

        if not webvtt_content or webvtt_content.startswith("Error"):
            return ""

        processed = process_webvtt_transcript(webvtt_content)
        return processed.get("transcript", "")

    def extract_segments(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract timestamped segments from a YouTube video.

        Args:
            url: The YouTube URL

        Returns:
            List of segments with text and timestamps
        """
        video_id, webvtt_content = extract_youtube_transcript(url)

        if not webvtt_content or webvtt_content.startswith("Error"):
            return None

        processed = process_webvtt_transcript(webvtt_content)
        segments = processed.get("segments", [])

        # Convert to standard format
        return [
            {
                "text": seg.get("content", ""),
                "start": seg.get("startSeconds", 0.0),
                "end": seg.get("endSeconds", 0.0),
            }
            for seg in segments
            if seg.get("content")
        ]

    def process(
        self,
        url: str,
        tags: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        skip_if_exists: bool = True,
    ) -> IngestionResult:
        """
        Process a YouTube video into the knowledge base.

        Args:
            url: The YouTube URL
            tags: Optional tags to apply
            collections: Optional collections to add to
            skip_if_exists: Skip processing if video already exists

        Returns:
            IngestionResult with success status and details
        """
        tags = tags or []
        collections = collections or []

        # Validate URL
        if not self.validate_url(url):
            return IngestionResult(
                success=False,
                error=f"Invalid YouTube URL: {url}",
            )

        try:
            # Extract video ID
            video_id = self._extract_video_id(url)
            if not video_id:
                return IngestionResult(
                    success=False,
                    error=f"Could not extract video ID from URL: {url}",
                )

            # Check if already exists
            if skip_if_exists and self._source_repo.exists(video_id):
                existing = self._source_repo.get(video_id)
                return IngestionResult(
                    success=True,
                    source=existing,
                    chunk_count=existing.chunk_count if existing else 0,
                    metadata={"skipped": True, "reason": "already_exists"},
                )

            # Extract metadata
            metadata = self.extract_metadata(url)
            if metadata.get('error'):
                return IngestionResult(
                    success=False,
                    error=f"Failed to extract metadata: {metadata['error']}",
                )

            # Extract transcript with segments
            _, webvtt_content = extract_youtube_transcript(url)
            if not webvtt_content or "Error" in webvtt_content or "No subtitle" in webvtt_content:
                return IngestionResult(
                    success=False,
                    error=f"Failed to extract transcript: {webvtt_content}",
                )

            processed = process_webvtt_transcript(webvtt_content)
            transcript = processed.get("transcript", "")
            segments = processed.get("segments", [])

            if not transcript or len(transcript) < 50:
                return IngestionResult(
                    success=False,
                    error="Transcript too short or empty",
                )

            # Create source
            source = Source(
                source_id=video_id,
                source_type="youtube",
                title=metadata.get("title", "Unknown"),
                channel=metadata.get("channel"),
                description=metadata.get("description"),
                url=f"https://youtube.com/watch?v={video_id}",
                metadata={
                    "duration": metadata.get("duration", 0),
                    "view_count": metadata.get("view_count", 0),
                    "upload_date": metadata.get("upload_date", ""),
                    "categories": metadata.get("categories", []),
                    "video_type": processed.get("metadata", {}).get("videoType", "general"),
                },
                tags=tags + metadata.get("tags", [])[:5],  # Include up to 5 YouTube tags
                collections=collections,
            )

            # Chunk the content with timestamps
            if segments:
                segment_dicts = [
                    {
                        "text": seg.get("content", ""),
                        "start": seg.get("startSeconds", 0.0),
                        "end": seg.get("endSeconds", 0.0),
                    }
                    for seg in segments
                    if seg.get("content")
                ]
                text_chunks = self._chunking_service.chunk_with_timestamps(segment_dicts)
            else:
                text_chunks = self._chunking_service.chunk_text(transcript)

            if not text_chunks:
                return IngestionResult(
                    success=False,
                    error="No chunks generated from transcript",
                )

            # Generate embeddings
            chunk_texts = [tc.content for tc in text_chunks]
            embeddings = self.embedding_provider.embed_documents(chunk_texts)

            # Create chunk models
            chunks = []
            for i, (text_chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                chunk = Chunk(
                    source_id=video_id,
                    content=text_chunk.content,
                    chunk_index=i,
                    vector=embedding,
                    timestamp_start=text_chunk.timestamp_start,
                    timestamp_end=text_chunk.timestamp_end,
                    source_type="youtube",
                    source_title=source.title,
                    source_channel=source.channel,
                    tags=source.tags,
                    collections=source.collections,
                    embedding_model=self.embedding_provider.model_name,
                )
                chunks.append(chunk)

            # Update source with processing info
            source.is_processed = True
            source.chunk_count = len(chunks)
            source.embedding_model = self.embedding_provider.model_name

            # Save to database
            self._source_repo.add(source)
            self._chunk_repo.add(chunks)

            return IngestionResult(
                success=True,
                source=source,
                chunk_count=len(chunks),
                metadata={
                    "video_type": processed.get("metadata", {}).get("videoType", "general"),
                    "transcript_length": len(transcript),
                    "segment_count": len(segments),
                },
            )

        except Exception as e:
            import traceback
            return IngestionResult(
                success=False,
                error=f"Processing failed: {str(e)}\n{traceback.format_exc()}",
            )

    def reprocess(
        self,
        source_id: str,
        force_reembed: bool = False,
    ) -> IngestionResult:
        """
        Reprocess an existing YouTube video.

        Useful for re-chunking or re-embedding with new settings.

        Args:
            source_id: The video ID to reprocess
            force_reembed: Whether to regenerate embeddings

        Returns:
            IngestionResult with success status
        """
        source = self._source_repo.get(source_id)
        if not source:
            return IngestionResult(
                success=False,
                error=f"Source not found: {source_id}",
            )

        # Delete existing chunks
        self._chunk_repo.delete_by_source(source_id)

        # Reprocess using the URL
        url = source.url or f"https://youtube.com/watch?v={source_id}"

        return self.process(
            url=url,
            tags=source.tags,
            collections=source.collections,
            skip_if_exists=False,
        )

    async def process_async(
        self,
        url: str,
        tags: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        skip_if_exists: bool = True,
    ) -> IngestionResult:
        """
        Async version of process - network I/O for YouTube download is the bottleneck.

        Args:
            url: The YouTube URL
            tags: Optional tags to apply
            collections: Optional collections to add to
            skip_if_exists: Skip processing if video already exists

        Returns:
            IngestionResult with success status and details
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.process(
                url=url,
                tags=tags,
                collections=collections,
                skip_if_exists=skip_if_exists,
            )
        )

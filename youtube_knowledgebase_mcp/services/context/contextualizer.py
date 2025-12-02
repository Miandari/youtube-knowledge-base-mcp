"""
Contextualization service with chaptering support for long videos.

Implements Anthropic's Contextual Retrieval pattern:
- Generate per-chunk context explaining its place in the document
- Optionally prepend a global document summary
- Handle long videos via chaptering strategy
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

from .provider import ContextProvider
from ..chunking import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """A chapter of a long video."""
    index: int
    start_char: int
    end_char: int
    content: str
    summary: Optional[str] = None


@dataclass
class ContextualizedChunk:
    """A chunk with generated contextual information."""
    original: TextChunk
    context: str           # Per-chunk context
    summary: Optional[str] # Global or chapter summary
    embedding_text: str    # Combined text for embedding
    chapter_index: Optional[int] = None  # For long videos


class ContextualizationService:
    """
    Service for generating contextual text for chunks.

    Implements two modes:
    - Standard mode: Uses full transcript for context (videos < 100k tokens)
    - Chaptering mode: Splits into 30-min chapters for very long videos

    Uses sequential processing to leverage OpenAI's prompt caching.
    """

    # Rough estimate: 1 token ≈ 4 characters
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        context_provider: ContextProvider,
        include_summary: bool = True,
        max_context_tokens: int = 100,
        summary_sentences: int = 3,
        max_transcript_tokens: int = 100000,
        chapter_duration_minutes: int = 30,
    ):
        """
        Initialize the contextualization service.

        Args:
            context_provider: The LLM provider for generating context
            include_summary: Whether to generate a global summary
            max_context_tokens: Max tokens for per-chunk context
            summary_sentences: Number of sentences for global summary
            max_transcript_tokens: Token threshold for triggering chaptering
            chapter_duration_minutes: Chapter duration for long videos
        """
        self.provider = context_provider
        self.include_summary = include_summary
        self.max_context_tokens = max_context_tokens
        self.summary_sentences = summary_sentences
        self.max_transcript_tokens = max_transcript_tokens
        self.chapter_duration_minutes = chapter_duration_minutes

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return len(text) // self.CHARS_PER_TOKEN

    def _needs_chaptering(self, transcript: str) -> bool:
        """Check if transcript exceeds token limit and needs chaptering."""
        estimated_tokens = self._estimate_tokens(transcript)
        needs_chapters = estimated_tokens >= self.max_transcript_tokens
        if needs_chapters:
            logger.info(
                f"Transcript ~{estimated_tokens} tokens exceeds {self.max_transcript_tokens}. "
                f"Using chaptering strategy."
            )
        return needs_chapters

    def _create_chapters(
        self,
        transcript: str,
        chunks: List[TextChunk],
        video_duration_seconds: Optional[float] = None,
    ) -> List[Chapter]:
        """
        Split transcript into chapters based on chunk timestamps or character positions.

        Strategy:
        1. If chunks have timestamps, use them to create time-based chapters
        2. Otherwise, split by character count to fit within token limits
        """
        chapter_duration_secs = self.chapter_duration_minutes * 60
        chapters = []

        # Check if we have timestamp info
        has_timestamps = any(c.timestamp_start is not None for c in chunks)

        if has_timestamps and video_duration_seconds:
            # Time-based chaptering
            num_chapters = max(1, int(video_duration_seconds / chapter_duration_secs) + 1)
            chapter_duration = video_duration_seconds / num_chapters

            for i in range(num_chapters):
                start_time = i * chapter_duration
                end_time = (i + 1) * chapter_duration

                # Find chunks in this time range
                chapter_chunks = [
                    c for c in chunks
                    if c.timestamp_start is not None
                    and start_time <= c.timestamp_start < end_time
                ]

                if chapter_chunks:
                    start_char = min(c.start_char for c in chapter_chunks)
                    end_char = max(c.end_char for c in chapter_chunks)
                    chapter_content = transcript[start_char:end_char]

                    chapters.append(Chapter(
                        index=i,
                        start_char=start_char,
                        end_char=end_char,
                        content=chapter_content,
                    ))
        else:
            # Character-based chaptering (fallback for content without timestamps)
            num_chapters = max(1, self._estimate_tokens(transcript) // self.max_transcript_tokens + 1)
            chars_per_chapter = len(transcript) // num_chapters

            start = 0
            chapter_idx = 0
            while start < len(transcript):
                end = min(start + chars_per_chapter, len(transcript))

                # Try to break at sentence boundary
                if end < len(transcript):
                    for delim in ['. ', '? ', '! ', '\n']:
                        pos = transcript.rfind(delim, start, end)
                        if pos > start:
                            end = pos + len(delim)
                            break

                chapters.append(Chapter(
                    index=chapter_idx,
                    start_char=start,
                    end_char=end,
                    content=transcript[start:end],
                ))
                chapter_idx += 1
                start = end

        logger.info(f"Created {len(chapters)} chapters for long video")
        return chapters

    def _get_chunk_chapter(self, chunk: TextChunk, chapters: List[Chapter]) -> Optional[Chapter]:
        """Find which chapter a chunk belongs to based on character position."""
        for chapter in chapters:
            if chapter.start_char <= chunk.start_char < chapter.end_char:
                return chapter
        # Fallback to last chapter if not found
        return chapters[-1] if chapters else None

    def contextualize_chunks(
        self,
        document: str,
        chunks: List[TextChunk],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ContextualizedChunk]:
        """
        Generate contextual text for each chunk.

        For short videos: Uses full transcript as context document
        For long videos: Uses chaptering strategy with chapter-specific context

        Args:
            document: The full transcript text
            chunks: List of TextChunks to contextualize
            metadata: Optional metadata (title, channel, duration)

        Returns:
            List of ContextualizedChunks with embedding_text ready for vectorization
        """
        if not chunks:
            return []

        metadata = metadata or {}
        video_duration = metadata.get("duration")  # seconds

        # Check if we need chaptering
        use_chapters = self._needs_chaptering(document)
        chapters: Optional[List[Chapter]] = None

        if use_chapters:
            # Create chapters and generate chapter summaries
            chapters = self._create_chapters(document, chunks, video_duration)
            for chapter in chapters:
                # Limit chapter content for summary generation
                summary = self.provider.generate_chapter_summary(
                    chapter.content[:50000],
                    chapter.index + 1,
                )
                chapter.summary = summary
                if summary:
                    logger.debug(f"Chapter {chapter.index + 1} summary: {summary[:80]}...")

        # Generate global video summary
        global_summary = None
        if self.include_summary:
            try:
                # For long videos, summarize first chapter + last chapter
                if use_chapters and chapters:
                    summary_text = chapters[0].content[:5000]
                    if len(chapters) > 1:
                        summary_text += "\n...\n" + chapters[-1].content[:5000]
                else:
                    # Use first 20k chars for summary (fits in context window)
                    summary_text = document[:20000]

                global_summary = self.provider.generate_summary(
                    summary_text,
                    self.summary_sentences,
                )
                if global_summary:
                    logger.info(f"Generated global summary: {global_summary[:100]}...")
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")

        # Build header from metadata
        header_parts = []
        if metadata.get("title"):
            header_parts.append(f"Video: {metadata['title']}")
        if metadata.get("channel"):
            header_parts.append(f"Channel: {metadata['channel']}")
        if global_summary:
            header_parts.append(f"Summary: {global_summary}")
        header = " | ".join(header_parts)

        # Generate per-chunk contexts (SEQUENTIAL for prompt caching)
        results = []
        first_context_logged = False

        for chunk in chunks:
            # Determine context document (full or chapter)
            if use_chapters and chapters:
                chapter = self._get_chunk_chapter(chunk, chapters)
                context_document = chapter.content if chapter else document[:50000]
                chapter_idx = chapter.index if chapter else None
                chapter_summary = chapter.summary if chapter else None
            else:
                context_document = document
                chapter_idx = None
                chapter_summary = None

            # Generate context for this chunk (sequential for caching)
            try:
                context = self.provider.generate_context(
                    document=context_document,
                    chunk=chunk.content,
                    max_tokens=self.max_context_tokens,
                )
                if context and not first_context_logged:
                    logger.info("First chunk context generated (prompt now cached)")
                    first_context_logged = True
            except Exception as e:
                logger.warning(f"Context generation failed for chunk {chunk.index}: {e}")
                context = ""

            # Build embedding text: header + chapter_summary + context + content
            parts = []
            if header:
                parts.append(header)
            if chapter_summary and use_chapters:
                parts.append(f"Chapter {chapter_idx + 1}: {chapter_summary}")
            if context:
                parts.append(context)
            parts.append(chunk.content)

            embedding_text = "\n\n".join(parts)

            results.append(ContextualizedChunk(
                original=chunk,
                context=context,
                summary=global_summary,
                embedding_text=embedding_text,
                chapter_index=chapter_idx,
            ))

        logger.info(f"Contextualized {len(results)} chunks")
        return results

    def contextualize_single(
        self,
        document: str,
        chunk: TextChunk,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextualizedChunk:
        """
        Contextualize a single chunk (for reprocessing).

        Args:
            document: The full transcript text
            chunk: The TextChunk to contextualize
            summary: Optional pre-generated summary
            metadata: Optional metadata (title, channel)

        Returns:
            ContextualizedChunk with embedding_text
        """
        metadata = metadata or {}

        # Generate context for this chunk
        try:
            # Limit document size for single chunk
            context = self.provider.generate_context(
                document=document[:100000],  # ~25k tokens max
                chunk=chunk.content,
                max_tokens=self.max_context_tokens,
            )
        except Exception as e:
            logger.warning(f"Context generation failed: {e}")
            context = ""

        # Build embedding text
        parts = []
        if metadata.get("title"):
            parts.append(f"Video: {metadata['title']}")
        if metadata.get("channel"):
            parts.append(f"Channel: {metadata['channel']}")
        if summary:
            parts.append(f"Summary: {summary}")
        if context:
            parts.append(context)
        parts.append(chunk.content)

        return ContextualizedChunk(
            original=chunk,
            context=context,
            summary=summary,
            embedding_text="\n\n".join(parts),
            chapter_index=None,
        )

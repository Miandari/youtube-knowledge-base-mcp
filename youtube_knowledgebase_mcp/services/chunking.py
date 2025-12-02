"""
Chunking service for splitting text into semantic chunks.

Implements sentence-boundary aware splitting with configurable chunk size and overlap.
Supports semantic chunking based on embedding similarity when an embedding provider is available.
"""
import re
from typing import List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from ..core.config import settings

if TYPE_CHECKING:
    from ..core.embeddings import EmbeddingProvider


@dataclass
class TextChunk:
    """A chunk of text with optional timestamp information."""
    content: str
    index: int
    start_char: int
    end_char: int
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None


class ChunkingService:
    """
    Service for splitting text into semantic chunks.

    Uses sentence-boundary aware splitting to avoid cutting mid-sentence.
    When an embedding provider is available, uses semantic similarity to detect
    topic shifts and create more coherent chunks.

    Configurable chunk size (~500 chars default) and overlap (150 chars default).
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_provider: Optional["EmbeddingProvider"] = None,
        context_window_size: int = 1,
    ):
        """
        Initialize the chunking service.

        Args:
            chunk_size: Target chunk size in characters (default from settings)
            chunk_overlap: Overlap between chunks in characters (default from settings)
            embedding_provider: Provider for generating embeddings (enables semantic chunking)
            context_window_size: Number of sentences before/after to include when embedding.
                                 Default 1 means 3 sentences total. Set to 0 to disable.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._embedding_provider = embedding_provider
        self.context_window_size = context_window_size

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.

        Handles common sentence-ending punctuation while avoiding
        false positives like "Mr." or "Dr."
        """
        # Simple sentence splitting on . ! ? followed by space or end
        # First, protect common abbreviations by replacing them temporarily
        protected = text
        abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'e.g.', 'i.e.']
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))

        # Split on sentence-ending punctuation followed by space
        pattern = r'[.!?]+\s+'
        sentences = re.split(pattern, protected)

        # Restore abbreviations and filter
        result = []
        for s in sentences:
            s = s.replace('<DOT>', '.').strip()
            if s:
                result.append(s)

        return result

    def _merge_short_sentences(
        self,
        sentences: List[str],
        min_length: int = 50
    ) -> List[str]:
        """
        Merge very short sentences with their neighbors.

        Args:
            sentences: List of sentences
            min_length: Minimum sentence length before merging

        Returns:
            List of sentences with short ones merged
        """
        if not sentences:
            return sentences

        merged = []
        buffer = ""

        for sentence in sentences:
            if buffer:
                buffer = buffer + " " + sentence
            else:
                buffer = sentence

            if len(buffer) >= min_length:
                merged.append(buffer)
                buffer = ""

        # Add any remaining buffer
        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        return merged

    def _combine_with_context(self, sentences: List[str]) -> List[str]:
        """
        Combine each sentence with neighbors for better embedding context.

        Uses self.context_window_size to determine how many sentences
        before/after to include. Default is 1 (3 sentences total).
        Set to 0 to disable context window.
        """
        if self.context_window_size == 0:
            return sentences

        combined = []
        window = self.context_window_size
        for i in range(len(sentences)):
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            combined.append(" ".join(sentences[start:end]))
        return combined

    def _calculate_cosine_distances(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine distance between consecutive embeddings."""
        distances = []
        for i in range(len(embeddings) - 1):
            vec_a = np.array(embeddings[i])
            vec_b = np.array(embeddings[i + 1])
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                distances.append(1.0)  # Max distance for zero vectors
            else:
                similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                distances.append(1 - similarity)  # Convert to distance
        return distances

    def _find_breakpoints(self, distances: List[float], percentile: int = 80) -> List[int]:
        """Find indices where semantic shift exceeds threshold."""
        if not distances:
            return []
        threshold = np.percentile(distances, percentile)
        return [i for i, d in enumerate(distances) if d > threshold]

    def chunk_semantically(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks based on semantic similarity.

        Detects topic shifts by comparing embedding similarity between
        consecutive sentences. Uses 80th percentile of distances as threshold.

        Falls back to chunk_text() if embedding provider is unavailable or fails.
        """
        if self._embedding_provider is None:
            return self.chunk_text(text)

        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return self.chunk_text(text)

        try:
            # Rolling window context for better embeddings
            combined = self._combine_with_context(sentences)

            # Batch embed all sentences (in chunks of 100 to avoid API limits)
            embeddings = []
            batch_size = 100
            for i in range(0, len(combined), batch_size):
                batch = combined[i:i + batch_size]
                embeddings.extend(self._embedding_provider.embed_documents(batch))

            # Calculate distances and find breakpoints
            distances = self._calculate_cosine_distances(embeddings)
            breakpoints = self._find_breakpoints(distances, percentile=80)

            # Group sentences into chunks
            chunks = []
            start_idx = 0
            char_offset = 0

            for bp in breakpoints:
                chunk_sentences = sentences[start_idx:bp + 1]
                chunk_text = " ".join(chunk_sentences)

                chunks.append(TextChunk(
                    content=chunk_text,
                    index=len(chunks),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                ))

                char_offset += len(chunk_text) + 1  # +1 for space
                start_idx = bp + 1

            # Final chunk
            if start_idx < len(sentences):
                chunk_sentences = sentences[start_idx:]
                chunk_text = " ".join(chunk_sentences)
                chunks.append(TextChunk(
                    content=chunk_text,
                    index=len(chunks),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                ))

            return chunks if chunks else self.chunk_text(text)

        except Exception:
            # Fallback on any embedding failure
            return self.chunk_text(text)

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks with sentence-boundary awareness.

        Args:
            text: The text to chunk

        Returns:
            List of TextChunks
        """
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self._split_into_sentences(text)
        sentences = self._merge_short_sentences(sentences)

        if not sentences:
            # If no sentences found, fall back to simple splitting
            return self._simple_chunk(text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_char = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                end_char = start_char + len(chunk_text)

                chunks.append(TextChunk(
                    content=chunk_text,
                    index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                ))
                chunk_index += 1

                # Calculate overlap - keep last sentences that fit in overlap
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_text = " ".join(overlap_sentences)
                    else:
                        break

                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_length = len(overlap_text)
                start_char = end_char - len(overlap_text)

            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                content=chunk_text,
                index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
            ))

        return chunks

    def _simple_chunk(self, text: str) -> List[TextChunk]:
        """
        Simple character-based chunking as fallback.

        Args:
            text: The text to chunk

        Returns:
            List of TextChunks
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at a space if not at the end
            if end < len(text):
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    content=chunk_text,
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                ))
                chunk_index += 1

            # Move start with overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            # Avoid infinite loop
            if start >= end:
                start = end

        return chunks

    def chunk_with_timestamps(
        self,
        segments: List[dict],
    ) -> List[TextChunk]:
        """
        Chunk text segments that have timestamp information.

        Preserves timestamp mapping from segments to chunks.

        Args:
            segments: List of dicts with 'text', 'start', 'end' keys

        Returns:
            List of TextChunks with timestamp information
        """
        if not segments:
            return []

        # Build full text and track segment positions
        full_text = ""
        segment_positions = []  # (start_char, end_char, start_time, end_time)

        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            start_char = len(full_text)
            if full_text:
                full_text += " "
                start_char += 1
            full_text += text
            end_char = len(full_text)

            segment_positions.append((
                start_char,
                end_char,
                seg.get("start", 0.0),
                seg.get("end", 0.0),
            ))

        # Use semantic chunking if provider available, else character-based
        if self._embedding_provider is not None:
            chunks = self.chunk_semantically(full_text)
        else:
            chunks = self.chunk_text(full_text)

        # Map timestamps to chunks
        for chunk in chunks:
            # Find segments that overlap with this chunk
            overlapping = [
                (start, end, ts_start, ts_end)
                for start, end, ts_start, ts_end in segment_positions
                if start < chunk.end_char and end > chunk.start_char
            ]

            if overlapping:
                chunk.timestamp_start = overlapping[0][2]  # First segment's start
                chunk.timestamp_end = overlapping[-1][3]   # Last segment's end

        return chunks

    def estimate_chunk_count(self, text: str) -> int:
        """
        Estimate the number of chunks for a given text.

        Args:
            text: The text to estimate

        Returns:
            Estimated number of chunks
        """
        if not text:
            return 0

        text_length = len(text)
        effective_chunk_size = self.chunk_size - self.chunk_overlap

        if effective_chunk_size <= 0:
            return 1

        return max(1, (text_length + effective_chunk_size - 1) // effective_chunk_size)

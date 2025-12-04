"""
Pydantic models for the knowledge base entities.
"""
from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel, Field
import uuid


class Source(BaseModel):
    """A source of knowledge (YouTube video, article, etc.)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str = "youtube"  # "youtube", "article", "podcast", etc.
    source_id: str  # video_id, url hash, etc.

    # Core metadata
    title: str
    channel: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None

    # Extended metadata (flexible JSON for source-type-specific data)
    # Example for YouTube: {"duration": 3600, "view_count": 1000000, "upload_date": "2024-01-15"}
    metadata: dict = Field(default_factory=dict)

    # Organization
    tags: List[str] = Field(default_factory=list)
    collections: List[str] = Field(default_factory=list)

    # User additions
    user_summary: Optional[str] = None

    # Processing state
    is_processed: bool = False
    chunk_count: int = 0
    embedding_model: Optional[str] = None

    # Timestamps - use datetime objects for PyArrow compatibility
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """A semantic chunk of content from a source."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str  # Foreign key to Source.source_id

    # Content
    content: str
    chunk_index: int

    # Vector (populated during embedding)
    vector: Optional[List[float]] = None

    # Timestamps (for time-based media)
    timestamp_start: Optional[float] = None  # seconds
    timestamp_end: Optional[float] = None

    # Denormalized source metadata (for efficient filtered search)
    source_type: str = "youtube"
    source_title: str = ""
    source_channel: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    collections: List[str] = Field(default_factory=list)

    # Embedding metadata
    embedding_model: str = ""

    # Contextual retrieval
    context: Optional[str] = None  # LLM-generated context for this chunk
    context_model: Optional[str] = None  # Model used for context generation

    # Future SOTA RAG fields
    parent_id: Optional[str] = None  # Small-to-Big Retrieval: link to parent chunk
    speakers: List[str] = Field(default_factory=list)  # Speaker diarization for filtering
    chapter_index: Optional[int] = None  # Chapter number for long videos

    # Timestamps - use datetime objects for PyArrow compatibility
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """A search result with relevance scoring."""
    chunk: Chunk
    score: float  # Raw similarity score
    recency_weight: float = 1.0
    final_score: float = 0.0  # score * recency_weight

    # Source info (denormalized for display)
    source_title: str
    source_url: Optional[str] = None
    timestamp_link: Optional[str] = None  # YouTube link with timestamp


# === MCP Tool Result Models ===

class ProcessResult(BaseModel):
    """Result of processing a source."""
    success: bool
    source_id: Optional[str] = None
    title: Optional[str] = None
    chunk_count: int = 0
    error: Optional[str] = None


class SearchResults(BaseModel):
    """Search results container."""
    query: str
    total_results: int
    results: List[SearchResult]
    search_time_ms: float = 0.0


class OperationResult(BaseModel):
    """Generic operation result."""
    success: bool
    message: str
    affected_count: int = 0


class StatsResult(BaseModel):
    """Knowledge base statistics."""
    total_sources: int
    total_chunks: int
    sources_by_type: Dict[str, int]
    unique_tags: int
    unique_collections: int
    embedding_model: str


class LibraryStats(BaseModel):
    """Library statistics for explore_library(view='stats')."""
    total_sources: int
    total_chunks: int
    sources_by_type: Dict[str, int]
    unique_tags: int
    tags: List[str]
    embedding_model: str
    data_path: str  # Enables LLM to guide users on data location

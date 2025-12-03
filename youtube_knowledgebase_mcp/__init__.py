"""
YouTube Knowledge Base MCP Module

This module provides tools for processing media transcripts, creating a searchable knowledge base,
and retrieving information from content based on natural language queries.

MCP Tools (4 consolidated, workflow-based tools):
- process_video: Ingest YouTube videos with optional tags/summary
- manage_source: Update tags and summaries
- explore_library: Browse sources, tags, and stats
- search: Semantic search

Admin operations are in the CLI (kb command).
"""

# Import the MCP instance and tools
from .mcp_tools import (
    mcp,
    process_video,
    manage_source,
    explore_library,
    search,
)

# Import core components
from .core import (
    settings,
    Settings,
    Source,
    Chunk,
    SearchResult,
    ProcessResult,
    SearchResults,
    OperationResult,
    StatsResult,
    get_db,
)
from .core.models import LibraryStats

# Import services
from .services import (
    SearchService,
    OrganizationService,
    ChunkingService,
)
from .services.ingestion import YouTubeIngestionService

# Import repositories
from .repositories import (
    SourceRepository,
    ChunkRepository,
)

# Legacy imports for backwards compatibility (will be removed in future)
from .youtube_transcript import (
    extract_youtube_transcript,
    process_webvtt_transcript,
    get_youtube_metadata,
)

# Define which symbols to export
__all__ = [
    # MCP
    'mcp',
    # Tools (4 consolidated)
    'process_video',
    'manage_source',
    'explore_library',
    'search',
    # Core
    'settings',
    'Settings',
    'Source',
    'Chunk',
    'SearchResult',
    'ProcessResult',
    'SearchResults',
    'OperationResult',
    'StatsResult',
    'LibraryStats',
    'get_db',
    # Services
    'SearchService',
    'OrganizationService',
    'ChunkingService',
    'YouTubeIngestionService',
    # Repositories
    'SourceRepository',
    'ChunkRepository',
    # Legacy
    'extract_youtube_transcript',
    'process_webvtt_transcript',
    'get_youtube_metadata',
]

# Define the version
__version__ = "0.1.0"

"""
Media Knowledge Base MCP Module

This module provides tools for processing media transcripts, creating a searchable knowledge base,
and retrieving information from content based on natural language queries.
"""

# Import the MCP instance and tools
from .mcp_tools import (
    mcp,
    process_youtube_video,
    search_knowledge_base,
    get_source,
    list_sources,
    add_tags,
    remove_tags,
    list_tags,
    add_to_collection,
    remove_from_collection,
    list_collections,
    set_summary,
    get_summary,
    get_status,
    reset_knowledge_base,
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
    # Tools
    'process_youtube_video',
    'search_knowledge_base',
    'get_source',
    'list_sources',
    'add_tags',
    'remove_tags',
    'list_tags',
    'add_to_collection',
    'remove_from_collection',
    'list_collections',
    'set_summary',
    'get_summary',
    'get_status',
    'reset_knowledge_base',
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

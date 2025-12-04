"""
MCP Tools - Consolidated workflow-based tools.

This module exposes 4 high-level MCP tools designed for LLM efficiency:
1. process_video - Ingest YouTube videos with optional tags/summary
2. manage_source - Update tags and summaries
3. explore_library - All metadata lookups (sources, tags, stats)
4. search - Semantic search

Administrative operations (reset, bulk import, etc.) are in cli.py.
"""
import time
from typing import List, Literal, Optional, Union

from mcp.server.fastmcp import FastMCP

from .core import settings
from .core.models import (
    Source,
    ProcessResult,
    SearchResults,
    LibraryStats,
)
from .services import SearchService, OrganizationService
from .services.ingestion import YouTubeIngestionService
from .repositories import SourceRepository

mcp = FastMCP("YouTube-KB")


# === Tool 1: process_video (The Ingestor) ===

@mcp.tool()
async def process_video(
    url: str,
    tags: Optional[List[str]] = None,
    summary: Optional[str] = None,
) -> ProcessResult:
    """
    Process a YouTube video and add it to the knowledge base.

    Extracts transcript, generates embeddings, and stores for semantic search.
    Optionally applies tags and summary in one atomic operation.
    Safe to call multiple times - existing videos are skipped.

    Args:
        url: YouTube video URL (any format: youtube.com, youtu.be, etc.)
        tags: Optional tags to apply immediately after ingestion
        summary: Optional user summary to set immediately after ingestion

    Returns:
        ProcessResult with success status, source_id, title, and chunk_count

    Examples:
        - process_video("https://youtube.com/watch?v=abc123")
        - process_video("https://youtu.be/abc123", tags=["python", "tutorial"])
        - process_video(url, tags=["ml"], summary="Great intro to neural networks")
    """
    try:
        service = YouTubeIngestionService()
        result = await service.process_async(url)

        if not result.success:
            return ProcessResult(success=False, error=result.error)

        source_id = result.source.source_id

        # Apply tags if provided
        if tags:
            OrganizationService().add_tags(source_id, tags)

        # Apply summary if provided
        if summary:
            OrganizationService().set_summary(source_id, summary)

        return ProcessResult(
            success=True,
            source_id=source_id,
            title=result.source.title,
            chunk_count=result.chunk_count,
        )
    except Exception as e:
        return ProcessResult(success=False, error=str(e))


# === Tool 2: manage_source (The Editor) ===

@mcp.tool()
def manage_source(
    source_id: str,
    add_tags: Optional[List[str]] = None,
    remove_tags: Optional[List[str]] = None,
    summary: Optional[str] = None,
) -> Source:
    """
    Update a source's tags and/or summary. Returns the updated source.

    This is the universal "edit" tool - modify tags, update summaries, or both
    in a single call. The updated source is returned so you can confirm changes
    without a separate lookup.

    Args:
        source_id: The source identifier (e.g., "dQw4w9WgXcQ")
        add_tags: Tags to add (idempotent - existing tags preserved)
        remove_tags: Tags to remove
        summary: New summary text. Use "" (empty string) to clear the summary.
                 Use None (omit) to leave unchanged.

    Returns:
        The updated Source object with all current metadata

    Examples:
        - manage_source("abc123", add_tags=["python"])
        - manage_source("abc123", remove_tags=["draft"], add_tags=["published"])
        - manage_source("abc123", summary="Key video about async programming")
        - manage_source("abc123", summary="")  # Clears the summary
    """
    org_service = OrganizationService()
    source_repo = SourceRepository()

    # Verify source exists
    source = source_repo.get(source_id)
    if source is None:
        raise ValueError(f"Source not found: {source_id}")

    # Apply tag additions
    if add_tags:
        org_service.add_tags(source_id, add_tags)

    # Apply tag removals
    if remove_tags:
        org_service.remove_tags(source_id, remove_tags)

    # Apply summary change (empty string clears, None leaves unchanged)
    if summary is not None:
        if summary == "":
            org_service.clear_summary(source_id)
        else:
            org_service.set_summary(source_id, summary)

    # Return updated source
    return source_repo.get(source_id)


# === Tool 3: explore_library (The Librarian) ===

@mcp.tool()
def explore_library(
    view: Literal["sources", "source", "tags", "stats"] = "sources",
    source_id: Optional[str] = None,
    filter_tags: Optional[List[str]] = None,
    filter_title: Optional[str] = None,
    sort_by: Literal["created_at", "updated_at", "title"] = "created_at",
    limit: int = 20,
) -> Union[List[Source], Source, List[str], LibraryStats]:
    """
    Explore and browse the knowledge base metadata.

    This is the universal "lookup" tool for browsing sources, viewing details,
    listing tags, or checking statistics. Use different 'view' modes for
    different types of lookups.

    Args:
        view: What to retrieve:
            - "sources": List sources matching filters (returns List[Source])
            - "source": Get one source by ID (returns Source, requires source_id)
            - "tags": List all unique tags (returns List[str])
            - "stats": Get library statistics (returns LibraryStats)
        source_id: Required when view="source". The source ID to look up.
        filter_tags: Filter sources by tags (sources with at least one matching tag)
        filter_title: Filter sources by title (case-insensitive substring)
        sort_by: Sort order for "sources" view
        limit: Max results for "sources" view (default 20)

    Returns:
        - view="sources": List[Source] - Sources matching filters
        - view="source": Source - Single source with full details including summary
        - view="tags": List[str] - All unique tags in the library
        - view="stats": LibraryStats - {total_sources, total_chunks, sources_by_type, unique_tags, tags, embedding_model}

    Examples:
        - explore_library()  # List recent sources
        - explore_library(view="source", source_id="abc123")  # Get one source
        - explore_library(filter_tags=["python"])  # Sources tagged "python"
        - explore_library(view="tags")  # List all tags
        - explore_library(view="stats")  # Get statistics
    """
    source_repo = SourceRepository()
    org_service = OrganizationService()

    if view == "source":
        if not source_id:
            raise ValueError("source_id is required when view='source'")
        source = source_repo.get(source_id)
        if source is None:
            raise ValueError(f"Source not found: {source_id}")
        return source

    elif view == "sources":
        return source_repo.list(
            tags=filter_tags,
            title_contains=filter_title,
            sort_by=sort_by,
            limit=limit,
        )

    elif view == "tags":
        return org_service.list_all_tags()

    elif view == "stats":
        stats = org_service.get_stats()
        return LibraryStats(
            total_sources=stats["total_sources"],
            total_chunks=stats["total_chunks"],
            sources_by_type=stats["sources_by_type"],
            unique_tags=stats["unique_tags"],
            tags=stats["tags"],
            embedding_model=settings.embedding.get_model_name(),
            data_path=str(settings.data_path),
        )

    else:
        raise ValueError(f"Invalid view: {view}. Must be one of: sources, source, tags, stats")


# === Tool 4: search (The Researcher) ===

@mcp.tool()
async def search(
    query: str,
    source_ids: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    limit: int = 10,
) -> SearchResults:
    """
    Search the knowledge base using hybrid semantic and keyword search.

    Uses a two-stage retrieval pipeline:
    1. Hybrid search combining vector similarity and full-text search
    2. Cross-encoder reranking for improved relevance
    3. HyDE query transformation for semantic bridging (if enabled)

    Results include relevance scores and YouTube timestamp links for navigation.

    Args:
        query: Natural language search query
        source_ids: Optional list of source IDs to search within
        tags: Optional list of tags to filter by
        limit: Maximum number of results (default 10)

    Returns:
        SearchResults with:
        - query: The original query
        - total_results: Number of results found
        - results: List of SearchResult objects with:
            - chunk: The matched content chunk
            - score: Relevance score
            - source_title: Title of the source
            - timestamp_link: Direct YouTube link with timestamp

    Examples:
        - search("how to use async await in Python")
        - search("machine learning basics", tags=["tutorial"])
        - search("error handling", source_ids=["abc123", "def456"])
    """
    start = time.time()
    service = SearchService()

    results = await service.search_async(
        query,
        limit=limit,
        source_ids=source_ids,
        tags=tags,
    )

    elapsed = (time.time() - start) * 1000

    return SearchResults(
        query=query,
        total_results=len(results),
        results=results,
        search_time_ms=elapsed,
    )

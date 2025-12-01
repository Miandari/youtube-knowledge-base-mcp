"""
MCP Tools - Thin wrappers around services.

This module defines all the MCP tools exposed through the MCP interface.
Each tool is a thin wrapper that delegates to the appropriate service.
"""
import time
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from .core import get_db, settings
from .core.models import (
    Source,
    ProcessResult,
    SearchResults,
    OperationResult,
    StatsResult,
)
from .services import SearchService, OrganizationService
from .services.ingestion import YouTubeIngestionService
from .repositories import SourceRepository

mcp = FastMCP("Media-KB-MCP")


# === Ingestion ===

@mcp.tool()
async def process_youtube_video(youtube_url: str) -> ProcessResult:
    """
    Process a YouTube video and add it to the knowledge base.

    Extracts transcript, generates embeddings, and stores for semantic search.
    Safe to call multiple times - existing videos are skipped.
    """
    try:
        service = YouTubeIngestionService()
        result = await service.process_async(youtube_url)
        if not result.success:
            return ProcessResult(success=False, error=result.error)
        return ProcessResult(
            success=True,
            source_id=result.source.source_id,
            title=result.source.title,
            chunk_count=result.chunk_count,
        )
    except Exception as e:
        return ProcessResult(success=False, error=str(e))


# === Search ===

@mcp.tool()
async def search_knowledge_base(
    query: str,
    source_ids: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    collections: Optional[List[str]] = None,
    limit: int = 10,
) -> SearchResults:
    """
    Search the knowledge base using hybrid semantic and keyword search.

    Results are ranked by relevance with a recency boost.
    Use filters to narrow results to specific sources, tags, or collections.
    """
    start = time.time()
    service = SearchService()
    results = await service.search_async(
        query,
        limit=limit,
        source_ids=source_ids,
        tags=tags,
        collections=collections,
    )
    elapsed = (time.time() - start) * 1000
    return SearchResults(
        query=query,
        total_results=len(results),
        results=results,
        search_time_ms=elapsed,
    )


# === Source Management ===

@mcp.tool()
def get_source(source_id: str) -> Source:
    """Get detailed information about a source."""
    source = SourceRepository().get(source_id)
    if source is None:
        raise ValueError(f"Source not found: {source_id}")
    return source


@mcp.tool()
def list_sources(
    source_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    collections: Optional[List[str]] = None,
    channel: Optional[str] = None,
    title_contains: Optional[str] = None,
    has_summary: Optional[bool] = None,
    limit: int = 50,
    sort_by: str = "created_at",
) -> List[Source]:
    """
    List sources with optional filtering.

    Args:
        source_type: Filter by type (e.g., "youtube")
        tags: Filter by tags (sources with at least one matching tag)
        collections: Filter by collections
        channel: Filter by channel name (exact match)
        title_contains: Filter by title (case-insensitive substring)
        has_summary: Filter by whether source has a user summary
        limit: Maximum number of results (default 50)
        sort_by: One of "created_at", "updated_at", "title"
    """
    return SourceRepository().list(
        source_type=source_type,
        tags=tags,
        collections=collections,
        channel=channel,
        title_contains=title_contains,
        has_summary=has_summary,
        limit=limit,
        sort_by=sort_by,
    )


# === Tags ===

@mcp.tool()
def add_tags(source_id: str, tags: List[str]) -> OperationResult:
    """Add tags to a source. Idempotent - existing tags are preserved."""
    source = OrganizationService().add_tags(source_id, tags)
    if source is None:
        return OperationResult(
            success=False, message=f"Source not found: {source_id}"
        )
    return OperationResult(
        success=True, message=f"Added tags: {tags}", affected_count=len(tags)
    )


@mcp.tool()
def remove_tags(source_id: str, tags: List[str]) -> OperationResult:
    """Remove tags from a source."""
    source = OrganizationService().remove_tags(source_id, tags)
    if source is None:
        return OperationResult(
            success=False, message=f"Source not found: {source_id}"
        )
    return OperationResult(
        success=True, message=f"Removed tags: {tags}", affected_count=len(tags)
    )


@mcp.tool()
def list_tags() -> List[str]:
    """List all unique tags across all sources."""
    return OrganizationService().list_all_tags()


# === Collections ===

@mcp.tool()
def add_to_collection(source_id: str, collection: str) -> OperationResult:
    """Add a source to a collection. Collections are auto-created."""
    source = OrganizationService().add_to_collection(source_id, collection)
    if source is None:
        return OperationResult(
            success=False, message=f"Source not found: {source_id}"
        )
    return OperationResult(
        success=True, message=f"Added to collection: {collection}"
    )


@mcp.tool()
def remove_from_collection(source_id: str, collection: str) -> OperationResult:
    """Remove a source from a collection."""
    source = OrganizationService().remove_from_collection(source_id, collection)
    if source is None:
        return OperationResult(
            success=False, message=f"Source not found: {source_id}"
        )
    return OperationResult(
        success=True, message=f"Removed from collection: {collection}"
    )


@mcp.tool()
def list_collections() -> List[str]:
    """List all unique collections across all sources."""
    return OrganizationService().list_all_collections()


# === Summaries ===

@mcp.tool()
def set_summary(source_id: str, summary: str) -> OperationResult:
    """Set or update the user summary for a source."""
    source = OrganizationService().set_summary(source_id, summary)
    if source is None:
        return OperationResult(
            success=False, message=f"Source not found: {source_id}"
        )
    return OperationResult(success=True, message="Summary updated")


@mcp.tool()
def get_summary(source_id: str) -> Optional[str]:
    """Get the user summary for a source."""
    return OrganizationService().get_summary(source_id)


# === Utilities ===

@mcp.tool()
def get_status() -> StatsResult:
    """Get knowledge base statistics."""
    stats = OrganizationService().get_stats()
    return StatsResult(
        total_sources=stats["total_sources"],
        total_chunks=stats["total_chunks"],
        sources_by_type=stats["sources_by_type"],
        unique_tags=stats["unique_tags"],
        unique_collections=stats["unique_collections"],
        embedding_model=settings.embedding.get_model_name(),
    )


@mcp.tool()
def reset_knowledge_base() -> OperationResult:
    """
    Reset the entire knowledge base. DESTRUCTIVE - deletes all data.

    Use with caution. This cannot be undone.
    """
    get_db().reset()
    return OperationResult(success=True, message="Knowledge base reset")

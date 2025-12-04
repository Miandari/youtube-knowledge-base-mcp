"""
Developer CLI for YouTube Knowledge Base administration.

This CLI provides administrative operations that should not be exposed
to LLMs via MCP, including destructive operations, bulk imports, and diagnostics.

Usage:
    uv run kb --help
    uv run kb db stats
    uv run kb db reset --confirm
    uv run kb source list
"""
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import click

from .core import get_db, settings
from .core.models import Source
from .repositories import SourceRepository, ChunkRepository
from .services import OrganizationService
from .services.ingestion import YouTubeIngestionService


@click.group()
@click.version_option(version="0.1.0", prog_name="kb")
def cli():
    """YouTube Knowledge Base - Developer CLI.

    Administrative tools for managing the knowledge base.
    For MCP tools exposed to LLMs, use the MCP server instead.
    """
    pass


# === Database Commands ===

@cli.group()
def db():
    """Database management commands."""
    pass


@db.command("stats")
def db_stats():
    """Show database statistics."""
    stats = OrganizationService().get_stats()

    click.echo("\n📊 Knowledge Base Statistics")
    click.echo("=" * 40)
    click.echo(f"Total sources:  {stats['total_sources']}")
    click.echo(f"Total chunks:   {stats['total_chunks']}")
    click.echo(f"Unique tags:    {stats['unique_tags']}")

    if stats['sources_by_type']:
        click.echo("\nSources by type:")
        for stype, count in stats['sources_by_type'].items():
            click.echo(f"  {stype}: {count}")

    if stats['tags']:
        click.echo(f"\nTags: {', '.join(stats['tags'][:10])}")
        if len(stats['tags']) > 10:
            click.echo(f"  ... and {len(stats['tags']) - 10} more")

    click.echo(f"\nEmbedding model: {settings.embedding.get_model_name()}")
    click.echo(f"Database path:   {settings.db_path}")
    click.echo()


@db.command("reset")
@click.option("--confirm", is_flag=True, help="Confirm destructive operation")
def db_reset(confirm: bool):
    """Reset the entire database. DESTRUCTIVE - deletes all data."""
    if not confirm:
        click.echo("⚠️  This will DELETE ALL DATA in the knowledge base.")
        click.echo("Run with --confirm to proceed.")
        sys.exit(1)

    click.echo("Resetting database...")
    get_db().reset()
    click.echo("✅ Database reset complete.")


@db.command("export")
@click.option("--format", "fmt", type=click.Choice(["json"]), default="json", help="Export format")
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
def db_export(fmt: str, output: Optional[str]):
    """Export all sources to JSON."""
    sources = SourceRepository().list(limit=10000)

    data = {
        "export_version": "1.0",
        "total_sources": len(sources),
        "sources": [s.model_dump(mode="json") for s in sources]
    }

    json_output = json.dumps(data, indent=2, default=str)

    if output:
        Path(output).write_text(json_output)
        click.echo(f"✅ Exported {len(sources)} sources to {output}")
    else:
        click.echo(json_output)


@db.command("migrate")
@click.argument("target_path", type=click.Path())
@click.option("--confirm", is_flag=True, help="Confirm the migration")
def db_migrate(target_path: str, confirm: bool):
    """Move the database to a new location safely."""
    current_path = settings.data_path
    target = Path(target_path).expanduser().resolve()

    if not current_path.exists():
        click.echo(f"❌ No database found at {current_path}")
        sys.exit(1)

    if target.exists():
        click.echo(f"❌ Target already exists: {target}")
        click.echo("   Aborting to prevent overwrite.")
        sys.exit(1)

    if not confirm:
        click.echo(f"📦 This will move your database:")
        click.echo(f"   From: {current_path}")
        click.echo(f"   To:   {target}")
        click.echo("\nRun with --confirm to proceed.")
        sys.exit(0)

    click.echo(f"📦 Moving data from {current_path} to {target}...")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(current_path), str(target))
        click.echo("✅ Migration successful!")
        click.echo("")
        click.echo("Next steps:")
        click.echo("1. Set this environment variable in your shell:")
        click.echo(f"   export YOUTUBE_KB_DATA_DIR={target}")
        click.echo("")
        click.echo("2. Or add to Claude Desktop config:")
        click.echo(f'   "env": {{ "YOUTUBE_KB_DATA_DIR": "{target}" }}')
        click.echo("")
        click.echo("3. Restart Claude Desktop")
    except Exception as e:
        click.echo(f"❌ Migration failed: {e}")
        sys.exit(1)


# === Source Commands ===

@cli.group()
def source():
    """Source management commands."""
    pass


@source.command("list")
@click.option("--tag", "-t", multiple=True, help="Filter by tag(s)")
@click.option("--type", "source_type", help="Filter by source type")
@click.option("--limit", "-n", default=20, help="Maximum results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def source_list(tag: tuple, source_type: Optional[str], limit: int, as_json: bool):
    """List sources with optional filtering."""
    sources = SourceRepository().list(
        tags=list(tag) if tag else None,
        source_type=source_type,
        limit=limit,
    )

    if as_json:
        click.echo(json.dumps([s.model_dump(mode="json") for s in sources], indent=2, default=str))
        return

    if not sources:
        click.echo("No sources found.")
        return

    click.echo(f"\n📚 Sources ({len(sources)} found)")
    click.echo("=" * 60)

    for s in sources:
        tags_str = f" [{', '.join(s.tags)}]" if s.tags else ""
        click.echo(f"• {s.source_id[:12]}... | {s.title[:40]:<40}{tags_str}")

    click.echo()


@source.command("get")
@click.argument("source_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def source_get(source_id: str, as_json: bool):
    """Get detailed information about a source."""
    source = SourceRepository().get(source_id)

    if not source:
        click.echo(f"❌ Source not found: {source_id}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(source.model_dump(mode="json"), indent=2, default=str))
        return

    click.echo(f"\n📄 Source: {source.title}")
    click.echo("=" * 60)
    click.echo(f"ID:          {source.source_id}")
    click.echo(f"Type:        {source.source_type}")
    click.echo(f"Channel:     {source.channel or 'N/A'}")
    click.echo(f"URL:         {source.url or 'N/A'}")
    click.echo(f"Tags:        {', '.join(source.tags) if source.tags else 'None'}")
    click.echo(f"Chunks:      {source.chunk_count}")
    click.echo(f"Processed:   {source.is_processed}")
    click.echo(f"Created:     {source.created_at}")

    if source.user_summary:
        click.echo(f"\nSummary:\n{source.user_summary}")

    click.echo()


@source.command("delete")
@click.argument("source_id")
@click.option("--confirm", is_flag=True, help="Confirm deletion")
def source_delete(source_id: str, confirm: bool):
    """Delete a source and all its chunks."""
    source = SourceRepository().get(source_id)

    if not source:
        click.echo(f"❌ Source not found: {source_id}", err=True)
        sys.exit(1)

    if not confirm:
        click.echo(f"⚠️  This will delete source '{source.title}' and {source.chunk_count} chunks.")
        click.echo("Run with --confirm to proceed.")
        sys.exit(1)

    # Delete chunks first, then source
    chunk_repo = ChunkRepository()
    deleted_chunks = chunk_repo.delete_by_source(source_id)
    SourceRepository().delete(source_id)

    click.echo(f"✅ Deleted source '{source.title}' and {deleted_chunks} chunks.")


# === Bulk Operations ===

@cli.command("bulk-tag")
@click.argument("file", type=click.Path(exists=True))
@click.argument("tags", nargs=-1)
def bulk_tag(file: str, tags: tuple):
    """Add tags to sources listed in a file (one source_id per line)."""
    if not tags:
        click.echo("❌ No tags specified.", err=True)
        sys.exit(1)

    source_ids = Path(file).read_text().strip().split("\n")
    source_ids = [sid.strip() for sid in source_ids if sid.strip()]

    org_service = OrganizationService()
    updated = org_service.bulk_add_tags(source_ids, list(tags))

    click.echo(f"✅ Added tags {list(tags)} to {updated}/{len(source_ids)} sources.")


@cli.command("import-urls")
@click.argument("file", type=click.Path(exists=True))
@click.option("--tag", "-t", multiple=True, help="Tags to apply to all imports")
def import_urls(file: str, tag: tuple):
    """Bulk import YouTube URLs from a file (one URL per line)."""
    urls = Path(file).read_text().strip().split("\n")
    urls = [url.strip() for url in urls if url.strip() and not url.startswith("#")]

    click.echo(f"📥 Importing {len(urls)} URLs...")

    service = YouTubeIngestionService()
    success_count = 0

    for i, url in enumerate(urls, 1):
        try:
            click.echo(f"  [{i}/{len(urls)}] {url[:60]}...", nl=False)
            result = asyncio.run(service.process_async(url))

            if result.success:
                # Add tags if specified
                if tag and result.source:
                    OrganizationService().add_tags(result.source.source_id, list(tag))
                click.echo(" ✅")
                success_count += 1
            else:
                click.echo(f" ❌ {result.error}")
        except Exception as e:
            click.echo(f" ❌ {e}")

    click.echo(f"\n✅ Imported {success_count}/{len(urls)} videos.")


# === Utility Commands ===

@cli.command("tags")
def list_tags():
    """List all unique tags."""
    tags = OrganizationService().list_all_tags()

    if not tags:
        click.echo("No tags found.")
        return

    click.echo(f"\n🏷️  Tags ({len(tags)} total)")
    click.echo("=" * 40)
    for tag in tags:
        click.echo(f"  • {tag}")
    click.echo()


@cli.command("config")
def show_config():
    """Show current configuration."""
    click.echo("\n⚙️  Configuration")
    click.echo("=" * 40)
    click.echo(f"Data directory:    {settings.data_path}")
    click.echo(f"Database path:     {settings.db_path}")
    click.echo(f"Embedding model:   {settings.embedding.get_model_name()}")
    click.echo(f"Embedding dims:    {settings.embedding.dimensions}")
    click.echo(f"Rerank enabled:    {settings.rerank.enabled}")
    click.echo(f"HyDE enabled:      {settings.hyde.enabled}")
    click.echo()
    click.echo("To change data location, set YOUTUBE_KB_DATA_DIR environment variable.")
    click.echo()


@cli.command("validate-url")
@click.argument("url")
def validate_url(url: str):
    """Validate a YouTube URL."""
    service = YouTubeIngestionService()

    if service.validate_url(url):
        video_id = service._extract_video_id(url)
        click.echo(f"✅ Valid YouTube URL")
        click.echo(f"   Video ID: {video_id}")
    else:
        click.echo(f"❌ Invalid YouTube URL: {url}")
        sys.exit(1)


@cli.command("health")
def health_check():
    """Run a full system health check."""
    click.echo("\n🏥 Health Check")
    click.echo("=" * 40)

    checks_passed = 0
    total_checks = 0

    # Check database connection
    total_checks += 1
    try:
        db = get_db()
        tables = db._db.table_names()
        click.echo(f"✅ Database connected ({len(tables)} tables)")
        checks_passed += 1
    except Exception as e:
        click.echo(f"❌ Database connection failed: {e}")

    # Check tables exist
    total_checks += 1
    try:
        db = get_db()
        if "sources" in db._db.table_names() and "chunks" in db._db.table_names():
            click.echo("✅ Required tables exist")
            checks_passed += 1
        else:
            click.echo("❌ Missing required tables")
    except Exception as e:
        click.echo(f"❌ Table check failed: {e}")

    # Check source/chunk counts
    total_checks += 1
    try:
        source_count = SourceRepository().count()
        chunk_count = ChunkRepository().count()
        click.echo(f"✅ Data accessible ({source_count} sources, {chunk_count} chunks)")
        checks_passed += 1
    except Exception as e:
        click.echo(f"❌ Data access failed: {e}")

    # Check embedding model config
    total_checks += 1
    try:
        model_name = settings.embedding.get_model_name()
        click.echo(f"✅ Embedding model configured: {model_name}")
        checks_passed += 1
    except Exception as e:
        click.echo(f"❌ Embedding config error: {e}")

    click.echo()
    if checks_passed == total_checks:
        click.echo(f"✅ All {total_checks} checks passed!")
    else:
        click.echo(f"⚠️  {checks_passed}/{total_checks} checks passed")
        sys.exit(1)


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()

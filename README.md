# YouTube Knowledge Base MCP

An MCP server that builds a searchable knowledge base from video content.

## Why

We consume more content than we can remember. Videos watched, podcasts heard, lectures attended—the information fades. This project builds a searchable knowledge base from that content. Start with YouTube, expand to other sources.

The key: it's an MCP server. Plug it into any LLM (Claude, GPT, local models) and your AI assistant can search everything you've ever watched. Your memory, augmented.

## Features

- Extract transcripts from YouTube videos
- Hybrid search (semantic + keyword)
- Timestamped links to exact video moments
- Organize with tags and notes
- Multiple embedding providers (Voyage, OpenAI, local)

## Installation

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- One of: Voyage API key, OpenAI API key, or local Ollama

### Setup

```bash
git clone https://github.com/yourusername/youtube-knowledge-base-mcp.git
cd youtube-knowledge-base-mcp
uv sync
```

### Environment

```bash
cp .env.example .env
```

Add your API key (at least one required):

```env
VOYAGE_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
```

## Usage

### With Claude Desktop (recommended)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "youtube-kb": {
      "command": "uv",
      "args": ["--directory", "/path/to/youtube-knowledge-base-mcp", "run", "youtube-kb"]
    }
  }
}
```

Then ask Claude: *"Add this video to my knowledge base: [URL]"*

### With Python

See `demo.ipynb` for interactive examples.

```python
from youtube_knowledgebase_mcp import process_video, search

# Add a video
result = await process_video("https://youtube.com/watch?v=...")

# Search
results = await search("What is context engineering?")
for r in results.results:
    print(r.timestamp_link)  # Jump to exact moment
```

## MCP Tools

4 workflow-based tools designed for LLM efficiency:

| Tool | Description |
|------|-------------|
| `process_video` | Add a video to the knowledge base (with optional tags/summary) |
| `manage_source` | Update tags and summary for a source |
| `explore_library` | Browse sources, list tags, or get statistics |
| `search` | Hybrid semantic + keyword search with reranking |

## Developer CLI

Administrative commands for database management (not exposed to LLMs):

```bash
uv run kb db stats           # Show database statistics
uv run kb db reset --confirm # Reset database (destructive)
uv run kb db migrate <path>  # Move database to new location
uv run kb source list        # List all sources
uv run kb source delete <id> # Delete a source
uv run kb health             # System health check
uv run kb import-urls <file> # Bulk import from file
```

Run `uv run kb --help` for all commands.

## Configuration

### Data Location

By default, data is stored in your OS's standard application data directory:
- **macOS**: `~/Library/Application Support/youtube-kb/`
- **Linux**: `~/.local/share/youtube-kb/`
- **Windows**: `%APPDATA%/youtube-kb/`

> **Note**: If you have existing data in `./data/` from a previous version, it will continue to be used automatically.

To use a custom location, set the `YOUTUBE_KB_DATA_DIR` environment variable:

```bash
export YOUTUBE_KB_DATA_DIR=/path/to/custom/location
```

Or in Claude Desktop config:

```json
{
  "mcpServers": {
    "youtube-kb": {
      "command": "uv",
      "args": ["--directory", "/path/to/repo", "run", "youtube-kb"],
      "env": {
        "YOUTUBE_KB_DATA_DIR": "/custom/data/path"
      }
    }
  }
}
```

### Moving Your Database

To move your database to a new location (e.g., Dropbox):

```bash
uv run kb db migrate ~/Dropbox/youtube-kb --confirm
```

Then follow the printed instructions to set the environment variable.

## Architecture

```
youtube_knowledgebase_mcp/
├── core/           # Config, models, database, embeddings
├── repositories/   # Data access layer (LanceDB)
├── services/       # Business logic (search, ingestion, organization)
├── mcp_tools.py    # MCP tools (4 workflow-based tools)
└── cli.py          # Developer CLI for admin operations
```

### Tech Stack

- **LanceDB** - Vector database with hybrid search
- **yt-dlp** - YouTube transcript extraction
- **Embeddings** - Voyage (default), OpenAI, BGE, Ollama
- **FastMCP** - MCP server framework

## License

MIT

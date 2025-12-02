# YouTube Knowledge Base MCP

An MCP server that builds a searchable knowledge base from video content.

## Why

We consume more content than we can remember. Videos watched, podcasts heard, lectures attended—the information fades. This project builds a searchable knowledge base from that content. Start with YouTube, expand to other sources.

The key: it's an MCP server. Plug it into any LLM (Claude, GPT, local models) and your AI assistant can search everything you've ever watched. Your memory, augmented.

## Features

- Extract transcripts from YouTube videos
- Hybrid search (semantic + keyword)
- Timestamped links to exact video moments
- Organize with tags, collections, and notes
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
from youtube_knowledgebase_mcp import process_youtube_video, search_knowledge_base

# Add a video
result = await process_youtube_video("https://youtube.com/watch?v=...")

# Search
results = await search_knowledge_base("What is context engineering?")
for r in results.results:
    print(r.timestamp_link)  # Jump to exact moment
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `process_youtube_video` | Add a video to the knowledge base |
| `search_knowledge_base` | Hybrid semantic + keyword search |
| `get_source` | Get video metadata |
| `list_sources` | List videos with filters |
| `add_tags` / `remove_tags` | Organize with tags |
| `add_to_collection` / `remove_from_collection` | Group into collections |
| `set_summary` / `get_summary` | Add personal notes |
| `list_tags` / `list_collections` | List all tags/collections |
| `get_status` | Knowledge base statistics |
| `reset_knowledge_base` | Clear all data |

## Architecture

```
youtube_knowledgebase_mcp/
├── core/           # Config, models, database, embeddings
├── repositories/   # Data access layer (LanceDB)
├── services/       # Business logic (search, ingestion, organization)
└── mcp_tools.py    # MCP tool definitions (thin wrappers)
```

### Tech Stack

- **LanceDB** - Vector database with hybrid search
- **yt-dlp** - YouTube transcript extraction
- **Embeddings** - Voyage (default), OpenAI, BGE, Ollama
- **FastMCP** - MCP server framework

## License

MIT

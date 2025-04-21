# MCP Server: YouTube Transcript Knowledge Base

**Ever find yourself struggling to recall insights from videos you've watched or wishing you didn't have to skim through hours of content just to find one piece of information?**

The **YouTube Transcript Knowledge Base** is an MCP (Model Context Protocol) server that lets you easily attach your favorite AI-powered tools (like LLMs) to a searchable knowledge base built from YouTube transcripts. This way, you can conveniently ask questions directly to your stored video content, helping you retain more information and search less.

## What is MCP?

**Model Context Protocol (MCP)** is an open standard enabling developers to securely connect their data sources with AI-powered tools. MCP servers expose your data, allowing any MCP-compatible AI tool (client) to access and interact with your datasets seamlessly. With MCP, integrating your data with AI becomes straightforward and highly efficient.

## Why You'll Love This

Imagine never having to rewatch an entire video to find a single quote or fact again. Instead, ask questions naturally through your preferred AI tool and instantly get precise answers. Ideal for students, educators, researchers, and lifelong learners who value their time and knowledge retention.

## Key Features

- **Transcript Extraction**: Automatically capture and process transcripts from YouTube videos.
- **Semantic Search**: Quickly find what you're looking for using semantic search.
- **Video Organization**: Keep your videos neatly organized in custom lists.
- **Personalized Summaries**: Add your notes and summaries for deeper understanding and faster review.
- **Intuitive Filtering**: Narrow down videos by channel, tags, view counts, and more.
- **Highlight Key Moments**: Automatically identify important segments within videos.
- **Prevent Information Overload**: Store knowledge effectively, making it easier to recall information.
- **Save Time**: Quickly retrieve specific information without scanning through entire videos.

## How Does It Work?

Here's the straightforward process:

1. **Capture**: Automatically grab video transcripts and metadata.
2. **Organize**: Segment and store transcripts semantically for easy retrieval.
3. **Query**: Connect your favorite LLM or AI-powered tool to ask natural language questions and receive accurate answers.
4. **Explore**: Jump directly to relevant timestamps for deeper context whenever needed.

### Tech Behind the Scenes

- **yt-dlp** for extracting transcripts and metadata.
- **FAISS** for efficient semantic searches.
- **OpenAI Embeddings** to relate content effectively.
- **LangChain** for streamlined document handling and processing.
- **uv** for fast, reliable dependency management.

## Getting Started

### Setup with uv (Quick & Easy)

This project uses [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager written in Rust that's 10-100x faster than pip.

Clone the repository and set up with uv:

```bash
# Install uv if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/youtube-transcript-knowledge-base.git
cd youtube-transcript-knowledge-base

# Create a virtual environment and install dependencies using uv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies from the lockfile for exact versions
uv sync

# Or install dependencies fresh from pyproject.toml
uv pip install -e .
```

Then, add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### No OpenAI API Key? No Problem!

Don't have an OpenAI API key? This project supports alternative embedding options:

- If you don't provide an OpenAI API key, the system will **automatically use Ollama** (a local embedding model) as a fallback
- You can also **explicitly choose** which embedding model to use in your code

#### Setting Up Ollama (for API-Free Usage)

To use Ollama embeddings (no API key required):

#### For macOS:
1. Download Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Install the application by opening the .dmg file and dragging Ollama to your Applications folder
3. Launch Ollama from your Applications folder
4. Pull the latest model (first time only):
   ```bash
   ollama pull llama3.2
   ```

#### For Linux:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service
ollama serve

# Download the latest language model
ollama pull llama3.2
```

#### For Windows:
1. Download Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Run the installer and follow the instructions
3. Launch Ollama from the Start menu
4. Pull the latest model (first time only):
   ```bash
   ollama pull llama3.2
   ```

#### Don't want to install Ollama? Use HuggingFace Instead

If you prefer not to install Ollama, the system will automatically fall back to using HuggingFace embeddings, which require no local installation or API key. HuggingFace embeddings work directly out of the box, though they may be slightly less performant than Ollama or OpenAI embeddings. The system handles this fallback transparently.

#### Choosing Your Embedding Type

```python
# In your code or notebook
from youtube_knowledgebase_mcp.vector_store import get_or_create_faiss_index

# Default (tries OpenAI first, then falls back to alternatives)
vectorstore = get_or_create_faiss_index(FAISS_INDEX_PATH)

# Explicitly choose Ollama (local, no API key needed)
vectorstore = get_or_create_faiss_index(FAISS_INDEX_PATH, embedding_type="ollama")

# Explicitly choose HuggingFace (no API key needed)
vectorstore = get_or_create_faiss_index(FAISS_INDEX_PATH, embedding_type="huggingface")

# Explicitly choose OpenAI (requires API key)
vectorstore = get_or_create_faiss_index(FAISS_INDEX_PATH, embedding_type="openai")
```

### Managing Dependencies with uv

This project uses `pyproject.toml` for dependency management with uv:

```bash
# Add a new dependency
uv add package-name

# Update the lockfile after modifying pyproject.toml
uv lock

# Sync your environment with the lockfile
uv sync
```

## Using Your Knowledge Base

### Run the MCP Server

```bash
# Standard way to run the server
uv run python main.py

# Alternatively, run the server from a specific directory
uv run --directory /address/to/folder/youtube_knowledgebase_mcp main.py
```

### Process New Videos

```bash
uv run python -c "from youtube_knowledgebase_mcp.mcp_tools import process_youtube_video; print(process_youtube_video('https://www.youtube.com/watch?v=VIDEO_ID'))"
```

### Ask Questions (Attach your favorite LLM)

```bash
uv run python -c "from youtube_knowledgebase_mcp.mcp_tools import youtube_transcript_query_tool; print(youtube_transcript_query_tool('your search query', 'VIDEO_ID'))"
```

### Organize Your Videos

```bash
uv run python -c "from youtube_knowledgebase_mcp.mcp_tools import create_video_list; print(create_video_list('Tech Reviews', 'Reviews of tech gadgets'))"
uv run python -c "from youtube_knowledgebase_mcp.mcp_tools import add_video_to_list; print(add_video_to_list('VIDEO_ID', 'Tech Reviews'))"
```

### Custom Summaries

```bash
uv run python -c "from youtube_knowledgebase_mcp.mcp_tools import add_video_summary; print(add_video_summary('VIDEO_ID', 'This video covers the key features of the latest iPhone.'))"
```

### Quick Filters

```bash
uv run python -c "from youtube_knowledgebase_mcp.mcp_tools import filter_videos; print(filter_videos(channel='Tech Channel'))"
```

## Connecting to Claude Desktop

One of the greatest advantages of MCP servers is their ability to connect with any compatible LLM, allowing you to leverage powerful AI capabilities to explore your knowledge base in natural language. Here's how to connect your YouTube Transcript Knowledge Base MCP server to Claude Desktop:

### Setting Up Claude Desktop for Your MCP Server

1. **Install Claude Desktop**: Make sure you have the latest version of [Claude for Desktop](https://claude.ai/desktop) installed on your computer.

2. **Configure Claude Desktop**: Open the Claude Desktop app configuration file:
   ```
   ~/Library/Application Support/Claude/claude_desktop_config.json
   ```
   Create this file if it doesn't exist.

3. **Add Your MCP Server**: Add your YouTube knowledge base server to the configuration file:
   ```json
   {
     "mcpServers": {
       "youtube-kb": {
         "command": "uv",
         "args": [
           "--directory",
           "/ABSOLUTE/PATH/TO/YouTube_MCP",
           "run",
           "main.py"
         ]
       }
     }
   }
   ```
   Replace `/ABSOLUTE/PATH/TO/YouTube_MCP` with the actual path to your project.

4. **Restart Claude Desktop**: Save the file and restart Claude Desktop for the changes to take effect.

### Using Your Knowledge Base with Claude

Once connected, you'll see a hammer icon in the Claude Desktop interface indicating that tools are available. You can now ask Claude questions about your YouTube videos:


- "Find information about machine learning from my tech tutorials playlist."
- "Show me key points about financial planning from my finance videos."

Claude will use your YouTube Knowledge Base MCP server to search through your transcripts and return relevant information along with timestamps and video sources.


## Project Structure

- `main.py`: Launches the MCP server.
- `mcp_tools.py`: Functions for interacting with your knowledge base.
- `youtube_transcript.py`: Handles transcript extraction.
- `vector_store.py`: FAISS vector database operations.
- `data_management.py`: Manages storage and retrieval of data files.
- `pyproject.toml`: Project configuration and dependencies for uv.
- `uv.lock`: Lockfile ensuring reproducible environments.

## Data Organization

Everything stored neatly:

- Transcripts: `data/processed_data/`
- Semantic Index: `data/youtube_faiss_index/`
- Video Lists: `data/video_lists.json`
- Summaries: `data/video_summaries.json`
- Metadata: `data/all_videos_metadata.json`

## Contribute & Grow

Contributions are welcomed—just open a Pull Request!

### Development Environment

Set up a development environment with uv:

```bash
# Create a development environment with additional tools
uv pip install -e ".[dev]"
```

## License

This project uses the MIT License. Check out `LICENSE` for details.

## Special Thanks

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for transcript extraction.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient searches.
- [LangChain](https://github.com/langchain-ai/langchain) for document handling.
- [OpenAI](https://openai.com/) for embeddings.
- [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.
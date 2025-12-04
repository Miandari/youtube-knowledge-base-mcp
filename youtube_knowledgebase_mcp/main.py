#!/usr/bin/env python3

"""
Media Knowledge Base MCP Server - Entry Point

This script starts the MCP server with all the tools for processing video transcripts,
building a searchable knowledge base, and managing content collections.
"""

import sys
import dotenv

# Import the MCP instance with all tools registered
from .mcp_tools import mcp

# Load environment variables (for API keys like VOYAGE_API_KEY)
dotenv.load_dotenv()


def main():
    """Main entry point for the Media Knowledge Base MCP Server"""
    # Use stderr for logging - stdout is reserved for MCP JSON-RPC protocol
    print("Starting Media Knowledge Base MCP Server", file=sys.stderr)
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
YouTube Knowledge Base MCP Server - Entry Point

This script starts the MCP server with all the tools for processing YouTube video transcripts,
building a searchable knowledge base, and managing video collections.
"""

import os
import dotenv
from mcp.server.fastmcp import FastMCP

# Import functions from other modules
from mcp_tools import init_mcp_tools, mcp

# Load environment variables
dotenv.load_dotenv()

# Define paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# print(f"Base path: {BASE_PATH}")
DATA_FOLDER = os.path.join(os.path.dirname(BASE_PATH), "data")
# print(f"Data folder: {DATA_FOLDER}")
DATA_PATH = os.path.join(DATA_FOLDER, "processed_data")
# print(f"Data path: {DATA_PATH}")
FAISS_INDEX_PATH = os.path.join(DATA_FOLDER, "youtube_faiss_index")
VIDEO_LISTS_PATH = os.path.join(DATA_FOLDER, "video_lists.json")
VIDEO_SUMMARIES_PATH = os.path.join(DATA_FOLDER, "video_summaries.json")
ALL_VIDEOS_METADATA_PATH = os.path.join(DATA_FOLDER, "all_videos_metadata.json")
# print(f"all videos metadata path: {ALL_VIDEOS_METADATA_PATH}")

# Create directories if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)

def main():
    """Main entry point for the YouTube Knowledgebase MCP Server"""
    # Create a dictionary of paths to pass to init_mcp_tools
    data_paths = {
        'DATA_PATH': DATA_PATH,
        'FAISS_INDEX_PATH': FAISS_INDEX_PATH,
        'VIDEO_LISTS_PATH': VIDEO_LISTS_PATH,
        'VIDEO_SUMMARIES_PATH': VIDEO_SUMMARIES_PATH,
        'ALL_VIDEOS_METADATA_PATH': ALL_VIDEOS_METADATA_PATH
    }
    
    # Initialize MCP tools with the path data
    init_mcp_tools(data_paths)
    
    # Start the MCP server
    print("Starting YouTube Knowledge Base MCP Server")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
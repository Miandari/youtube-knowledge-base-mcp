"""
YouTube Knowledgebase MCP Module

This module provides tools for processing YouTube transcripts, creating a searchable knowledge base,
and retrieving information from videos based on natural language queries.
"""

import os
from pathlib import Path

# Default paths
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_FAISS_INDEX_PATH = os.path.join(DEFAULT_DATA_PATH, "youtube_faiss_index")
DEFAULT_VIDEO_LISTS_PATH = os.path.join(DEFAULT_DATA_PATH, "video_lists.json")
DEFAULT_VIDEO_SUMMARIES_PATH = os.path.join(DEFAULT_DATA_PATH, "video_summaries.json")
DEFAULT_ALL_VIDEOS_METADATA_PATH = os.path.join(DEFAULT_DATA_PATH, "all_videos_metadata.json")

# Make sure paths exist
os.makedirs(DEFAULT_DATA_PATH, exist_ok=True)
os.makedirs(DEFAULT_FAISS_INDEX_PATH, exist_ok=True)

# Define key path variables for external access
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = DEFAULT_DATA_PATH
FAISS_INDEX_PATH = DEFAULT_FAISS_INDEX_PATH
VIDEO_LISTS_PATH = DEFAULT_VIDEO_LISTS_PATH
VIDEO_SUMMARIES_PATH = DEFAULT_VIDEO_SUMMARIES_PATH
ALL_VIDEOS_METADATA_PATH = DEFAULT_ALL_VIDEOS_METADATA_PATH
try:
# Import mcp_tools module to initialize the MCP server and register tools
    from .mcp_tools import (
        mcp, 
        init_mcp_tools,
        process_transcript_from_file,
        youtube_transcript_query_tool,
        check_knowledge_base_status,
        reset_knowledge_base_tool,
        process_youtube_video,
        add_video_to_list,
        remove_video_from_list,
        create_video_list,
        delete_video_list,
        get_video_lists,
        add_video_summary,
        get_video_summary
    )
except ImportError as e:
        from .mcp_tools import (
        mcp, 
        init_mcp_tools,
        process_transcript_from_file,
        youtube_transcript_query_tool,
        check_knowledge_base_status,
        reset_knowledge_base_tool,
        process_youtube_video,
        add_video_to_list,
        remove_video_from_list,
        create_video_list,
        delete_video_list,
        get_video_lists,
        add_video_summary,
        get_video_summary
    )

# Initialize the tool paths (these can be overridden by applications)
init_mcp_tools({
    'DATA_PATH': DEFAULT_DATA_PATH,
    'FAISS_INDEX_PATH': DEFAULT_FAISS_INDEX_PATH,
    'VIDEO_LISTS_PATH': DEFAULT_VIDEO_LISTS_PATH,
    'VIDEO_SUMMARIES_PATH': DEFAULT_VIDEO_SUMMARIES_PATH,
    'ALL_VIDEOS_METADATA_PATH': DEFAULT_ALL_VIDEOS_METADATA_PATH
})

# Import key modules and components for convenience
from .youtube_transcript import (
    extract_youtube_transcript, 
    process_webvtt_transcript,
    get_youtube_metadata
)
from .vector_store import (
    get_or_create_faiss_index,
    reset_knowledge_base
)
from .data_management import (
    load_video_lists,
    save_video_lists,
    load_video_summaries,
    save_video_summaries,
    load_all_videos_metadata,
    save_all_videos_metadata
)

# Define which symbols to export
__all__ = [
    'mcp',
    'init_mcp_tools',
    'process_transcript_from_file',
    'youtube_transcript_query_tool',
    'check_knowledge_base_status',
    'reset_knowledge_base_tool',
    'process_youtube_video',
    'add_video_to_list',
    'remove_video_from_list',
    'create_video_list',
    'delete_video_list',
    'get_video_lists',
    'add_video_summary',
    'get_video_summary',
    'BASE_PATH',
    'DATA_PATH',
    'FAISS_INDEX_PATH',
    'VIDEO_LISTS_PATH',
    'VIDEO_SUMMARIES_PATH',
    'ALL_VIDEOS_METADATA_PATH',
    'extract_youtube_transcript',
    'process_webvtt_transcript',
    'get_youtube_metadata',
    'get_or_create_faiss_index',
    'reset_knowledge_base',
    'load_video_lists',
    'save_video_lists',
    'load_video_summaries',
    'save_video_summaries',
    'load_all_videos_metadata',
    'save_all_videos_metadata'
]

# Define the version
__version__ = "0.1.0"
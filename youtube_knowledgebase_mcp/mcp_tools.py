"""
MCP Tools Module

This module defines all the MCP tools that will be exposed to the user through the MCP interface.
"""

import os
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
from mcp.server.fastmcp import FastMCP
try:
    # Imports from our custom modules
    from .youtube_transcript import (
        extract_youtube_transcript, 
        process_webvtt_transcript,
        load_webvtt_file,
        get_youtube_metadata
    )
    from .vector_store import (
        create_semantic_chunks, 
        get_or_create_faiss_index, 
        reset_knowledge_base,
        update_faiss_with_summary
    )
    from .data_management import (
        load_video_lists,
        save_video_lists,
        load_video_summaries,
        save_video_summaries,
        load_all_videos_metadata,
        save_all_videos_metadata,
        save_processed_data
    )
except ImportError as e:

    # Imports from our custom modules
    from youtube_transcript import (
        extract_youtube_transcript, 
        process_webvtt_transcript,
        load_webvtt_file,
        get_youtube_metadata
    )
    from vector_store import (
        create_semantic_chunks, 
        get_or_create_faiss_index, 
        reset_knowledge_base,
        update_faiss_with_summary
    )
    from data_management import (
        load_video_lists,
        save_video_lists,
        load_video_summaries,
        save_video_summaries,
        load_all_videos_metadata,
        save_all_videos_metadata,
        save_processed_data
    )
# Create a single MCP instance that will be used by all tools
mcp = FastMCP("YouTube-Knowledgebase-MCP")

# Global variables to store path information
DATA_PATH = None
FAISS_INDEX_PATH = None
VIDEO_LISTS_PATH = None
VIDEO_SUMMARIES_PATH = None
ALL_VIDEOS_METADATA_PATH = None

def init_mcp_tools(data_paths=None):
    """
    Initialize the MCP tools with the necessary path information.
    
    Args:
        data_paths (dict): Dictionary containing path information
    """
    global DATA_PATH, FAISS_INDEX_PATH, VIDEO_LISTS_PATH, VIDEO_SUMMARIES_PATH, ALL_VIDEOS_METADATA_PATH
    
    # Initialize paths if provided
    if data_paths:
        DATA_PATH = data_paths.get('DATA_PATH')
        FAISS_INDEX_PATH = data_paths.get('FAISS_INDEX_PATH')
        VIDEO_LISTS_PATH = data_paths.get('VIDEO_LISTS_PATH')
        VIDEO_SUMMARIES_PATH = data_paths.get('VIDEO_SUMMARIES_PATH')
        ALL_VIDEOS_METADATA_PATH = data_paths.get('ALL_VIDEOS_METADATA_PATH')

@mcp.tool()
def process_transcript_from_file(file_path: str, video_id: str) -> str:
    """
    Load a transcript file, process it, and save the results.
    
    Args:
        file_path (str): Path to WebVTT file
        video_id (str): Video ID or identifier
        
    Returns:
        str: Status message
    """
    try:
        # Load the WebVTT file
        webvtt_content = load_webvtt_file(file_path)
        
        # Process the transcript
        processed_data = process_webvtt_transcript(webvtt_content)
        
        # Save the processed data
        output_file = os.path.join(DATA_PATH, f"{video_id}_processed.json")
        save_result = save_processed_data(processed_data, output_file)
        
        return f"Successfully processed transcript for video {video_id}: {save_result}"
    except Exception as e:
        return f"Error processing transcript: {str(e)}"

@mcp.tool()
def youtube_transcript_query_tool(query: str, goal: Optional[str] = None, video_ids: Optional[List[str]] = None, 
                                metadata_filter: Optional[Dict[str, Any]] = None) -> str:
    """
    Query the YouTube transcript knowledge base for relevant information.
    
    Args:
        query (str): The query to search the YouTube transcripts with
        goal (str, optional): The user's goal or intent (e.g., "Find financial charts")
        video_ids (List[str], optional): Limit search to specific videos
        metadata_filter (Dict[str, Any], optional): Filter by metadata fields (e.g. {"channel": "Finance Channel"})

    Returns:
        str: A formatted string of the retrieved transcript segments with metadata information
    """
    try:
        # Check if FAISS index exists
        if not os.path.exists(FAISS_INDEX_PATH):
            return "Knowledge base hasn't been created yet. Add transcripts first."
            
        # Create the retriever
        vectorstore = get_or_create_faiss_index(FAISS_INDEX_PATH)
        
        # Enhance query with goal if provided
        enhanced_query = query
        if goal:
            enhanced_query = f"{query} {goal}"
        
        # Construct search parameters
        search_kwargs = {"k": 5}
        
        # Apply filter if video_ids or metadata_filter provided
        if video_ids or metadata_filter:
            def filter_func(doc):
                # Filter by video_ids if provided
                if video_ids and doc.metadata.get("video_id") not in video_ids:
                    return False
                
                # Filter by metadata if provided
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        # Handle list values (like tags, categories)
                        if isinstance(value, list):
                            doc_value = doc.metadata.get(key, [])
                            if not any(item in doc_value for item in value):
                                return False
                        # Handle scalar values
                        elif doc.metadata.get(key) != value:
                            return False
                
                return True
            
            retriever = vectorstore.as_retriever(
                search_kwargs=search_kwargs,
                search_type="similarity",
                filter=filter_func
            )
        else:
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # Retrieve relevant documents
        relevant_docs = retriever.invoke(enhanced_query)
        
        # Filter out initialization document
        relevant_docs = [doc for doc in relevant_docs if doc.metadata.get("video_id") != "init"]
        
        print(f"Retrieved {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            return "No relevant information found. Try a different query or add more transcripts to the knowledge base."
        
        # Format the response
        response_parts = []
        for i, doc in enumerate(relevant_docs):
            video_id = doc.metadata.get("video_id", "unknown")
            start_time = doc.metadata.get("start_time", "00:00:00")
            title = doc.metadata.get("title", "Unknown Title")
            channel = doc.metadata.get("channel", "Unknown Channel")
            
            response_parts.append(f"==SEGMENT {i+1} ==")
            response_parts.append(f"Video: {title}")
            response_parts.append(f"Channel: {channel}")
            response_parts.append(f"ID: {video_id}, Time: {start_time}")
            response_parts.append(f"\nContent: {doc.page_content}")
            
            # Include screenshot information if available
            screenshots = doc.metadata.get("screenshots", [])
            if screenshots:
                response_parts.append("\nRelevant Screenshot Moments:")
                for j, screenshot in enumerate(screenshots):
                    keywords = screenshot.get('keywords', [])
                    keyword_str = ', '.join(keywords) if keywords else "N/A"
                    response_parts.append(f"  - {screenshot['time']} - Keywords: {keyword_str}")
            
            # Include additional metadata if available
            if "tags" in doc.metadata and doc.metadata["tags"]:
                response_parts.append(f"\nTags: {', '.join(doc.metadata['tags'][:5])}" + 
                                    ("..." if len(doc.metadata["tags"]) > 5 else ""))
            
            if "categories" in doc.metadata and doc.metadata["categories"]:
                response_parts.append(f"Categories: {', '.join(doc.metadata['categories'])}")
                
            if "view_count" in doc.metadata:
                response_parts.append(f"Views: {doc.metadata['view_count']}")
                
            response_parts.append("")  # Empty line between segments
        
        formatted_response = "\n".join(response_parts)
        return formatted_response
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error querying knowledge base: {str(e)}\n\nTraceback:\n{tb}"
    
@mcp.tool()
def check_knowledge_base_status() -> str:
    """
    Check the status of the knowledge base.
    
    Returns:
        str: Information about the knowledge base status
    """
    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            return "Knowledge base doesn't exist yet. Add transcripts first."
        
        # Load the FAISS index
        vectorstore = get_or_create_faiss_index(FAISS_INDEX_PATH)
        
        # Get information about processed transcripts
        processed_files = [f for f in os.listdir(DATA_PATH) if f.endswith('_processed.json')]
        video_ids = [f.replace('_processed.json', '') for f in processed_files]
        
        # Try to get document count (not directly available in FAISS)
        index_size = "Unknown (FAISS doesn't expose document count directly)"
        try:
            # Try to estimate by checking the docstore
            import pickle
            docstore_path = os.path.join(FAISS_INDEX_PATH, "docstore.pkl")
            if os.path.exists(docstore_path):
                with open(docstore_path, "rb") as f:
                    docstore = pickle.load(f)
                    index_size = f"Approximately {len(docstore._dict)} documents"
        except Exception as e:
            print(f"Error estimating index size: {str(e)}")
        
        # Basic information
        info = [
            f"Knowledge base location: {FAISS_INDEX_PATH}",
            f"Index size: {index_size}",
            f"Number of processed videos: {len(processed_files)}",
            f"Processed video IDs: {', '.join(video_ids) if len(video_ids) <= 10 else ', '.join(video_ids[:10]) + '... (and more)'}"
        ]
        
        return "\n".join(info)
    except Exception as e:
        return f"Error checking knowledge base: {str(e)}"

@mcp.tool()
def reset_knowledge_base_tool() -> str:
    """
    Reset the knowledge base by deleting the FAISS index.
    
    Returns:
        str: Status message
    """
    return reset_knowledge_base(FAISS_INDEX_PATH)

@mcp.tool()
def process_youtube_video(youtube_url: str, query: Optional[str] = None) -> str:
    """
    All-in-one function to process a YouTube video and optionally query its content.
    This function handles the entire workflow in one step: extracting subtitles, 
    processing them, and adding them to the knowledge base.
    
    Args:
        youtube_url (str): YouTube video URL to process
        query (str, optional): Optional query to run against the transcript after processing
        
    Returns:
        str: Processing results and optional query results
    """
    try:
        # Step 1: Extract video metadata to get the ID
        print(f"Getting metadata for: {youtube_url}")
        metadata = get_youtube_metadata(youtube_url)
        video_id = metadata.get('video_id', '')
        video_title = metadata.get('title', 'Unknown title')
        
        if not video_id:
            return f"Failed to extract video ID from URL: {youtube_url}"
        
        # Load the centralized metadata
        all_metadata = load_all_videos_metadata(ALL_VIDEOS_METADATA_PATH)
            
        # Check if this video has already been processed
        processed_file = os.path.join(DATA_PATH, f"{video_id}_processed.json")
        
        if os.path.exists(processed_file) and video_id in all_metadata:
            print(f"Video '{video_title}' (ID: {video_id}) is already in the database")
            
            # If a query was provided, we can still run it against the existing data
            if query:
                print(f"Running query: '{query}' against existing video: {video_id}")
                query_result = youtube_transcript_query_tool(query, video_ids=[video_id])
                return (
                    f"Video '{video_title}' (ID: {video_id}) was already processed previously.\n\n"
                    f"Query Results for '{query}':\n"
                    f"{query_result}"
                )
            else:
                return f"Video '{video_title}' (ID: {video_id}) is already in the database. No need to process it again."
        
        # If we get here, the video hasn't been processed yet
        print(f"Processing video: '{video_title}'")
        
        # Step 2: Extract transcript
        print(f"Extracting transcript from YouTube")
        video_id, webvtt_content = extract_youtube_transcript(youtube_url)
        
        if not video_id:
            return f"Failed to extract video ID from URL: {youtube_url}"
        
        if not webvtt_content or len(webvtt_content) < 100 or webvtt_content.startswith("Error") or webvtt_content.startswith("No "):
            return f"Failed to extract valid transcript for video '{video_title}' (ID: {video_id})"
        
        print(f"Successfully extracted transcript for video ID: {video_id}")
        
        # Step 3: Process the transcript
        print(f"Processing transcript content...")
        processed_data = process_webvtt_transcript(webvtt_content)
        
        # Step 4: Save processed data
        output_file = os.path.join(DATA_PATH, f"{video_id}_processed.json")
        save_processed_data(processed_data, output_file)
        print(f"Saved processed transcript data to {output_file}")
        
        # Step 5: Save metadata to centralized storage
        all_metadata[video_id] = metadata
        save_all_videos_metadata(all_metadata, ALL_VIDEOS_METADATA_PATH)
        print(f"Saved video metadata to centralized storage")
        
        # Step 6: Create semantic chunks and add to knowledge base
        print(f"Creating semantic chunks and updating knowledge base...")
        # Pass the metadata to create_semantic_chunks
        documents = create_semantic_chunks(processed_data, video_id, metadata)
        
        if not documents:
            return f"Warning: No documents were created from the transcript for video '{video_title}' (ID: {video_id})"
        
        # Add documents to FAISS index
        vectorstore = get_or_create_faiss_index(FAISS_INDEX_PATH, documents)
        
        processing_result = f"Successfully processed YouTube video '{video_title}' (ID: {video_id}) with {len(documents)} semantic chunks"
        print(processing_result)
        
        # If no query was provided, return just the processing result
        if not query:
            return processing_result
        
        # Step 7: Query the processed transcript if requested
        print(f"Running query: '{query}' against video: {video_id}")
        query_result = youtube_transcript_query_tool(query, video_ids=[video_id])
        
        # Return combined results
        return (
            f"{processing_result}\n\n"
            f"Query Results for '{query}':\n"
            f"{query_result}"
        )
        
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error in all-in-one processing: {str(e)}\n\nTraceback:\n{tb}"

@mcp.tool()
def add_video_to_list(video_id: str, list_name: str = "default") -> str:
    """
    Add a video to a specified list for organization and filtering.
    
    Args:
        video_id (str): The YouTube video ID
        list_name (str): Name of the list to add the video to (default: "default")
        
    Returns:
        str: Status message
    """
    try:
        # Check if the video exists in our system
        processed_file = os.path.join(DATA_PATH, f"{video_id}_processed.json")
        
        # Load the centralized metadata
        all_metadata = load_all_videos_metadata(ALL_VIDEOS_METADATA_PATH)
        
        if not os.path.exists(processed_file) or video_id not in all_metadata:
            return f"Error: Video {video_id} has not been processed yet. Process the video first."
            
        # Get metadata from centralized storage
        metadata = all_metadata[video_id]
        video_title = metadata.get('title', 'Unknown')
        
        # Load video lists
        video_lists = load_video_lists(VIDEO_LISTS_PATH)
        
        # Create the list if it doesn't exist
        if list_name not in video_lists:
            video_lists[list_name] = {
                "description": f"List created for {video_title}",
                "video_ids": []
            }
            
        # Add video ID to the list if not already there
        if video_id not in video_lists[list_name]["video_ids"]:
            video_lists[list_name]["video_ids"].append(video_id)
            save_video_lists(video_lists, VIDEO_LISTS_PATH)
            return f"Added video '{video_title}' to list '{list_name}'"
        else:
            return f"Video '{video_title}' is already in list '{list_name}'"
            
    except Exception as e:
        return f"Error adding video to list: {str(e)}"

@mcp.tool()
def remove_video_from_list(video_id: str, list_name: str) -> str:
    """
    Remove a video from a specified list.
    
    Args:
        video_id (str): The YouTube video ID
        list_name (str): Name of the list to remove the video from
        
    Returns:
        str: Status message
    """
    try:
        # Load video lists
        video_lists = load_video_lists(VIDEO_LISTS_PATH)
        
        # Check if the list exists
        if list_name not in video_lists:
            return f"Error: List '{list_name}' does not exist"
            
        # Load the centralized metadata
        all_metadata = load_all_videos_metadata(ALL_VIDEOS_METADATA_PATH)
        
        # Get video title from centralized metadata if available
        video_title = video_id  # Default to ID
        if video_id in all_metadata:
            metadata = all_metadata[video_id]
            video_title = metadata.get('title', video_id)
            
        # Remove video ID from the list if it's there
        if video_id in video_lists[list_name]["video_ids"]:
            video_lists[list_name]["video_ids"].remove(video_id)
            save_video_lists(video_lists, VIDEO_LISTS_PATH)
            return f"Removed video '{video_title}' from list '{list_name}'"
        else:
            return f"Video '{video_title}' is not in list '{list_name}'"
            
    except Exception as e:
        return f"Error removing video from list: {str(e)}"

@mcp.tool()
def create_video_list(list_name: str, description: str) -> str:
    """
    Create a new list for organizing videos.
    
    Args:
        list_name (str): Name for the new list
        description (str): Description of what this list contains
        
    Returns:
        str: Status message
    """
    try:
        # Load video lists
        video_lists = load_video_lists(VIDEO_LISTS_PATH)
        
        # Check if the list already exists
        if list_name in video_lists:
            return f"Error: List '{list_name}' already exists"
            
        # Create the new list
        video_lists[list_name] = {
            "description": description,
            "video_ids": []
        }
        
        # Save updated lists
        save_video_lists(video_lists, VIDEO_LISTS_PATH)
        return f"Created new list '{list_name}': {description}"
            
    except Exception as e:
        return f"Error creating video list: {str(e)}"

@mcp.tool()
def delete_video_list(list_name: str) -> str:
    """
    Delete a video list. This does not delete the videos, just the list.
    
    Args:
        list_name (str): Name of the list to delete
        
    Returns:
        str: Status message
    """
    try:
        # Load video lists
        video_lists = load_video_lists(VIDEO_LISTS_PATH)
        
        # Don't allow deleting the default list
        if list_name == "default":
            return "Cannot delete the default list"
            
        # Check if the list exists
        if list_name not in video_lists:
            return f"Error: List '{list_name}' does not exist"
            
        # Delete the list
        del video_lists[list_name]
        
        # Save updated lists
        save_video_lists(video_lists, VIDEO_LISTS_PATH)
        return f"Deleted list '{list_name}'"
            
    except Exception as e:
        return f"Error deleting video list: {str(e)}"

@mcp.tool()
def get_video_lists() -> str:
    """
    Get all video lists with their contents.
    
    Returns:
        str: Formatted list of all video lists and their videos
    """
    try:
        # Load video lists
        video_lists = load_video_lists(VIDEO_LISTS_PATH)
        
        if not video_lists:
            return "No video lists found"
            
        # Load centralized metadata
        all_metadata = load_all_videos_metadata(ALL_VIDEOS_METADATA_PATH)
            
        # Format the response
        response_parts = ["=== Video Lists ==="]
        
        for list_name, list_info in video_lists.items():
            video_ids = list_info.get("video_ids", [])
            response_parts.append(f"\n## {list_name} ({len(video_ids)} videos)")
            response_parts.append(f"Description: {list_info.get('description', 'No description')}")
            
            if video_ids:
                response_parts.append("\nVideos:")
                for i, video_id in enumerate(video_ids, 1):
                    # Get the video title from centralized metadata
                    if video_id in all_metadata:
                        metadata = all_metadata[video_id]
                        video_title = metadata.get('title', video_id)
                    else:
                        video_title = video_id
                        
                    response_parts.append(f"{i}. {video_title} (ID: {video_id})")
            else:
                response_parts.append("No videos in this list")
                
            response_parts.append("")  # Empty line between lists
        
        return "\n".join(response_parts)
            
    except Exception as e:
        return f"Error getting video lists: {str(e)}"

@mcp.tool()
def add_video_summary(video_id: str, summary: str, add_to_search_index: bool = True) -> str:
    """
    Add a custom summary to a video that can be searched later.
    
    Args:
        video_id (str): The YouTube video ID
        summary (str): Your custom summary of the video
        add_to_search_index (bool): Whether to add this summary to the search index
        
    Returns:
        str: Status message
    """
    try:
        # Check if the video exists in our system
        processed_file = os.path.join(DATA_PATH, f"{video_id}_processed.json")
        
        # Load the centralized metadata
        all_metadata = load_all_videos_metadata(ALL_VIDEOS_METADATA_PATH)
        
        if not os.path.exists(processed_file) or video_id not in all_metadata:
            return f"Error: Video {video_id} has not been processed yet. Process the video first."
            
        # Get metadata from centralized storage
        metadata = all_metadata[video_id]
        video_title = metadata.get('title', 'Unknown')
        
        # Load existing summaries
        summaries = load_video_summaries(VIDEO_SUMMARIES_PATH)
        
        # Add or update the summary
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        summaries[video_id] = {
            "title": video_title,
            "summary": summary,
            "timestamp": timestamp
        }
        
        # Save the summaries
        save_video_summaries(summaries, VIDEO_SUMMARIES_PATH)
        
        # Add to search index if requested
        if add_to_search_index:
            update_faiss_with_summary(video_id, summary, DATA_PATH, FAISS_INDEX_PATH)
            status = "Added to search index and saved"
        else:
            status = "Saved (not added to search index)"
            
        return f"Summary for '{video_title}' {status}:\n\n{summary}"
            
    except Exception as e:
        return f"Error adding video summary: {str(e)}"

@mcp.tool()
def get_video_summary(video_id: str) -> str:
    """
    Get the custom summary for a specific video.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        str: The video summary or error message
    """
    try:
        # Load summaries
        summaries = load_video_summaries(VIDEO_SUMMARIES_PATH)
        
        # Check if this video has a summary
        if video_id not in summaries:
            # Try to get the video title
            metadata_file = os.path.join(DATA_PATH, f"{video_id}_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    import json
                    metadata = json.load(f)
                    video_title = metadata.get('title', video_id)
                return f"No summary found for video '{video_title}' (ID: {video_id})"
            else:
                return f"No summary found for video ID: {video_id}"
        
        # Format the response
        summary_info = summaries[video_id]
        response = [
            f"=== Summary for '{summary_info.get('title', video_id)}' ===",
            f"Last updated: {summary_info.get('timestamp', 'Unknown')}",
            "",
            summary_info.get('summary', 'No summary content')
        ]
        
        return "\n".join(response)
            
    except Exception as e:
        return f"Error getting video summary: {str(e)}"

def search_videos_by_list(query: str, list_name: str = None) -> str:
    """
    Search for videos, optionally restricting to a specific list.
    
    Args:
        query (str): The search query
        list_name (str, optional): Restrict search to this list name
        
    Returns:
        str: Search results
    """
    try:
        # Load video lists if we need to filter by list
        video_ids = None
        if list_name:
            video_lists = load_video_lists(VIDEO_LISTS_PATH)
            
            if list_name not in video_lists:
                return f"Error: List '{list_name}' does not exist"
                
            video_ids = video_lists[list_name].get("video_ids", [])
            
            if not video_ids:
                return f"List '{list_name}' is empty. No videos to search."
        
        # Search with optional list filter
        return youtube_transcript_query_tool(
            query=query,
            video_ids=video_ids,
            metadata_filter={"is_summary": True} if "summary" in query.lower() else None
        )
            
    except Exception as e:
        return f"Error searching videos: {str(e)}"
    
@mcp.tool()
def get_all_videos_info() -> str:
    """
    Get comprehensive information about all videos in the knowledge base.
    
    Returns:
        str: Formatted information about all videos including ID, title, channel, etc.
    """
    try:
        # Load centralized metadata instead of scanning directory
        all_metadata = load_all_videos_metadata(ALL_VIDEOS_METADATA_PATH)
        
        if not all_metadata:
            return "No videos have been processed yet in the knowledge base."
        
        video_ids = list(all_metadata.keys())
        
        # Load video lists for grouping information
        video_lists = load_video_lists(VIDEO_LISTS_PATH)
        
        # Create a mapping of video IDs to their list memberships
        video_list_membership = {}
        for list_name, list_info in video_lists.items():
            for vid_id in list_info.get("video_ids", []):
                if vid_id not in video_list_membership:
                    video_list_membership[vid_id] = []
                video_list_membership[vid_id].append(list_name)
        
        # Load summaries
        summaries = load_video_summaries(VIDEO_SUMMARIES_PATH)
        
        # Build comprehensive information about each video
        response_parts = ["=== All Videos in Knowledge Base ===\n"]
        
        for video_id in sorted(video_ids):
            # Get metadata from centralized storage
            metadata = all_metadata[video_id]
            
            video_title = metadata.get('title', 'Unknown')
            channel = metadata.get('channel', 'Unknown')
            upload_date = metadata.get('upload_date', 'Unknown')
            view_count = metadata.get('view_count', 'Unknown')
            
            # Format the upload date if available
            if upload_date and len(upload_date) == 8:
                upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
            
            response_parts.append(f"ID: {video_id}")
            response_parts.append(f"Title: {video_title}")
            response_parts.append(f"Channel: {channel}")
            response_parts.append(f"Upload Date: {upload_date}")
            response_parts.append(f"Views: {view_count:,}" if isinstance(view_count, int) else f"Views: {view_count}")
            
            # Add list membership information
            lists = video_list_membership.get(video_id, [])
            if lists:
                response_parts.append(f"Lists: {', '.join(lists)}")
            else:
                response_parts.append("Lists: None")
            
            # Add summary status
            has_summary = video_id in summaries
            response_parts.append(f"Has Custom Summary: {'Yes' if has_summary else 'No'}")
            
            response_parts.append("")  # Empty line between videos
        
        return "\n".join(response_parts)
            
    except Exception as e:
        return f"Error getting videos information: {str(e)}"
    
@mcp.tool()
def get_video_info(video_id: str) -> str:
    """
    Get detailed information about a specific video.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        str: Detailed information about the video
    """
    try:
        # Check if the video exists in our system
        processed_file = os.path.join(DATA_PATH, f"{video_id}_processed.json")
        
        # Load the centralized metadata
        all_metadata = load_all_videos_metadata(ALL_VIDEOS_METADATA_PATH)
        
        if not os.path.exists(processed_file) or video_id not in all_metadata:
            return f"Video {video_id} has not been processed yet or doesn't exist in the knowledge base."
            
        # Get metadata from centralized storage
        metadata = all_metadata[video_id]
        
        # Load processed data to get transcript length
        with open(processed_file, 'r', encoding='utf-8') as f:
            import json
            processed_data = json.load(f)
            
        # Load video lists to find which lists contain this video
        video_lists = load_video_lists(VIDEO_LISTS_PATH)
        containing_lists = []
        
        for list_name, list_info in video_lists.items():
            if video_id in list_info.get("video_ids", []):
                containing_lists.append(list_name)
                
        # Check if there's a custom summary
        summaries = load_video_summaries(VIDEO_SUMMARIES_PATH)
        summary_info = summaries.get(video_id, {})
        
        # Build the response
        response_parts = [f"=== Video Information: {metadata.get('title', 'Unknown')} ===\n"]
        
        # Basic info
        response_parts.append(f"Video ID: {video_id}")
        response_parts.append(f"Title: {metadata.get('title', 'Unknown')}")
        response_parts.append(f"Channel: {metadata.get('channel', 'Unknown')}")
        
        # Format upload date if available
        upload_date = metadata.get('upload_date', 'Unknown')
        if upload_date and len(upload_date) == 8:
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        response_parts.append(f"Upload Date: {upload_date}")
        
        # Additional metadata
        view_count = metadata.get('view_count', 'Unknown')
        response_parts.append(f"Views: {view_count:,}" if isinstance(view_count, int) else f"Views: {view_count}")
        
        duration = metadata.get('duration', 0)
        if duration:
            minutes, seconds = divmod(duration, 60)
            hours, minutes = divmod(minutes, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
            response_parts.append(f"Duration: {duration_str}")
        
        # Categories and tags
        categories = metadata.get('categories', [])
        if categories:
            response_parts.append(f"Categories: {', '.join(categories)}")
            
        tags = metadata.get('tags', [])
        if tags:
            tags_str = ', '.join(tags[:10])
            if len(tags) > 10:
                tags_str += f", ... (+{len(tags) - 10} more)"
            response_parts.append(f"Tags: {tags_str}")
        
        # List membership
        if containing_lists:
            response_parts.append(f"\nIncluded in lists: {', '.join(containing_lists)}")
        else:
            response_parts.append("\nNot included in any lists")
            
        # Transcript information
        transcript_length = len(processed_data.get('transcript', ''))
        num_segments = len(processed_data.get('segments', []))
        response_parts.append(f"\nTranscript: {transcript_length} characters, {num_segments} segments")
        
        # Video type from analysis
        video_type = processed_data.get('metadata', {}).get('videoType', 'Unknown')
        response_parts.append(f"Detected video type: {video_type}")
        
        # Key moments
        key_moments = processed_data.get('keyMoments', [])
        if key_moments:
            response_parts.append(f"Key moments for screenshots: {len(key_moments)}")
        
        # Summary information
        if summary_info:
            response_parts.append(f"\nCustom summary available (added {summary_info.get('timestamp', 'Unknown')})")
            response_parts.append("\nSummary excerpt: " + summary_info.get('summary', 'No summary')[:150] + "..." if len(summary_info.get('summary', '')) > 150 else "")
        else:
            response_parts.append("\nNo custom summary available")
            
        # YouTube link
        response_parts.append(f"\nYouTube URL: https://www.youtube.com/watch?v={video_id}")
        
        return "\n".join(response_parts)
            
    except Exception as e:
        return f"Error getting video information: {str(e)}"
    
@mcp.tool()
def filter_videos(list_name: str = None, category: str = None, 
                channel: str = None, tag: str = None, keyword_in_title: str = None,
                keyword_in_summary: str = None, has_summary: bool = None,
                min_view_count: int = None, max_videos: int = None,
                sort_by: str = "upload_date") -> str:
    """
    Get a filtered subset of videos based on various criteria.
    
    Args:
        list_name (str, optional): Filter to videos in this list
        category (str, optional): Filter to videos with this category
        channel (str, optional): Filter to videos from this channel
        tag (str, optional): Filter to videos with this tag
        keyword_in_title (str, optional): Filter to videos with this keyword in title
        keyword_in_summary (str, optional): Filter to videos with this keyword in their summary
        has_summary (bool, optional): Filter to videos that have (or don't have) a custom summary
        min_view_count (int, optional): Filter to videos with at least this many views
        max_videos (int, optional): Maximum number of videos to return (default: all)
        sort_by (str, optional): Sort videos by: "upload_date", "view_count", "title" (default: upload_date)
        
    Returns:
        str: Formatted information about the filtered videos
    """
    try:
        # Get list of all processed video files
        processed_files = [f for f in os.listdir(DATA_PATH) if f.endswith('_processed.json')]
        video_ids = [f.replace('_processed.json', '') for f in processed_files]
        
        if not video_ids:
            return "No videos have been processed yet in the knowledge base."
        
        # Load video lists if needed
        video_lists = {}
        if list_name:
            video_lists = load_video_lists(VIDEO_LISTS_PATH)
            if list_name not in video_lists:
                return f"Error: List '{list_name}' does not exist"
            video_ids = [vid for vid in video_ids if vid in video_lists[list_name].get("video_ids", [])]
        
        # Load summaries if needed
        summaries = {}
        if keyword_in_summary is not None or has_summary is not None:
            summaries = load_video_summaries(VIDEO_SUMMARIES_PATH)
            if has_summary is not None:
                video_ids = [vid for vid in video_ids if (vid in summaries) == has_summary]
            if keyword_in_summary:
                video_ids = [vid for vid in video_ids if vid in summaries and 
                        keyword_in_summary.lower() in summaries[vid].get('summary', '').lower()]
        
        # Initialize filtered videos data
        filtered_videos = []
        
        # Process each video
        for video_id in video_ids:
            # Get metadata
            metadata_file = os.path.join(DATA_PATH, f"{video_id}_metadata.json")
            if not os.path.exists(metadata_file):
                continue
                
            with open(metadata_file, 'r', encoding='utf-8') as f:
                import json
                metadata = json.load(f)
            
            # Apply filters
            if category and category.lower() not in [c.lower() for c in metadata.get('categories', [])]:
                continue
                
            if channel and channel.lower() not in metadata.get('channel', '').lower():
                continue
                
            if tag and not any(tag.lower() in t.lower() for t in metadata.get('tags', [])):
                continue
                
            if keyword_in_title and keyword_in_title.lower() not in metadata.get('title', '').lower():
                continue
                
            if min_view_count is not None and metadata.get('view_count', 0) < min_view_count:
                continue
            
            # Add video to filtered list with key metadata
            video_data = {
                'id': video_id,
                'title': metadata.get('title', 'Unknown'),
                'channel': metadata.get('channel', 'Unknown'),
                'upload_date': metadata.get('upload_date', ''),
                'view_count': metadata.get('view_count', 0),
                'duration': metadata.get('duration', 0),
                'categories': metadata.get('categories', []),
                'has_summary': video_id in summaries
            }
            
            filtered_videos.append(video_data)
        
        # Handle empty results
        if not filtered_videos:
            return "No videos match the specified filters."
            
        # Sort results
        if sort_by == "upload_date":
            filtered_videos.sort(key=lambda x: x.get('upload_date', ''), reverse=True)
        elif sort_by == "view_count":
            filtered_videos.sort(key=lambda x: x.get('view_count', 0), reverse=True)
        elif sort_by == "title":
            filtered_videos.sort(key=lambda x: x.get('title', '').lower())
        
        # Apply max_videos limit
        if max_videos is not None and max_videos > 0:
            filtered_videos = filtered_videos[:max_videos]
            
        # Format the response
        response_parts = [f"=== Filtered Videos ({len(filtered_videos)} results) ===\n"]
        
        # Add filter description
        filters_applied = []
        if list_name:
            filters_applied.append(f"List: {list_name}")
        if category:
            filters_applied.append(f"Category: {category}")
        if channel:
            filters_applied.append(f"Channel: {channel}")
        if tag:
            filters_applied.append(f"Tag: {tag}")
        if keyword_in_title:
            filters_applied.append(f"Title contains: {keyword_in_title}")
        if keyword_in_summary:
            filters_applied.append(f"Summary contains: {keyword_in_summary}")
        if has_summary is not None:
            filters_applied.append(f"Has summary: {has_summary}")
        if min_view_count is not None:
            filters_applied.append(f"Min views: {min_view_count:,}")
            
        if filters_applied:
            response_parts.append(f"Filters: {', '.join(filters_applied)}")
            response_parts.append(f"Sort: {sort_by}")
            response_parts.append("")
        
        # Format each video
        for i, video in enumerate(filtered_videos, 1):
            # Format the upload date if available
            upload_date = video.get('upload_date', '')
            if upload_date and len(upload_date) == 8:
                upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                
            # Format duration
            duration_sec = video.get('duration', 0)
            if duration_sec > 0:
                minutes, seconds = divmod(duration_sec, 60)
                hours, minutes = divmod(minutes, 60)
                duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
            else:
                duration = "Unknown"
                
            response_parts.append(f"{i}. {video['title']}")
            response_parts.append(f"   ID: {video['id']} | Channel: {video['channel']}")
            response_parts.append(f"   Date: {upload_date} | Duration: {duration} | Views: {video['view_count']:,}" if isinstance(video['view_count'], int) else f"   Date: {upload_date} | Duration: {duration} | Views: {video['view_count']}")
            
            if video['categories']:
                response_parts.append(f"   Categories: {', '.join(video['categories'][:3])}" + 
                                    ("..." if len(video['categories']) > 3 else ""))
                
            response_parts.append(f"   Has Summary: {'Yes' if video['has_summary'] else 'No'}")
            response_parts.append("")  # Empty line between videos
        
        # Add direct search helper suggestion
        response_parts.append("Use get_video_info(video_id) for detailed information on any video")
        
        return "\n".join(response_parts)
            
    except Exception as e:
        return f"Error filtering videos: {str(e)}"


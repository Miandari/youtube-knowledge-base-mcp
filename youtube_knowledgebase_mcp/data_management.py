"""
Data Management Module

This module handles all data management functionality including:
- Video lists management
- Video summaries management
- Metadata management
"""

import os
import json
from typing import Dict, Any, List, Optional

def initialize_data_files(data_paths: Dict[str, str]) -> None:
    """
    Initialize all data files if they don't exist
    
    Args:
        data_paths (Dict[str, str]): Dictionary containing all necessary path information
    """
    # Extract paths from the dictionary
    data_path = data_paths.get('DATA_PATH')
    video_lists_path = data_paths.get('VIDEO_LISTS_PATH')
    video_summaries_path = data_paths.get('VIDEO_SUMMARIES_PATH')
    all_videos_metadata_path = data_paths.get('ALL_VIDEOS_METADATA_PATH')
    
    # Create directories if they don't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Initialize all data files if they don't exist
    if not os.path.exists(video_lists_path):
        with open(video_lists_path, 'w', encoding='utf-8') as f:
            json.dump({
                "default": {
                    "description": "Default video list",
                    "video_ids": []
                }
            }, f, indent=2)

    if not os.path.exists(video_summaries_path):
        with open(video_summaries_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)
            
    if not os.path.exists(all_videos_metadata_path):
        with open(all_videos_metadata_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)

# Legacy version of the function for backward compatibility
def initialize_data_files_legacy(data_path: str, 
                          video_lists_path: str, 
                          video_summaries_path: str, 
                          all_videos_metadata_path: str) -> None:
    """
    Legacy version of initialize_data_files for backward compatibility
    
    Use initialize_data_files with a dictionary of paths instead.
    """
    # Create data paths dictionary
    data_paths = {
        'DATA_PATH': data_path,
        'VIDEO_LISTS_PATH': video_lists_path,
        'VIDEO_SUMMARIES_PATH': video_summaries_path,
        'ALL_VIDEOS_METADATA_PATH': all_videos_metadata_path
    }
    
    # Call the new version of the function
    initialize_data_files(data_paths)

def load_video_lists(video_lists_path: str) -> Dict[str, Dict[str, Any]]:
    """Load video lists from storage"""
    try:
        with open(video_lists_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupt or missing, create a new one with default list
        default_lists = {
            "default": {
                "description": "Default video list",
                "video_ids": []
            }
        }
        with open(video_lists_path, 'w', encoding='utf-8') as f:
            json.dump(default_lists, f, indent=2)
        return default_lists

def save_video_lists(lists: Dict[str, Dict[str, Any]], video_lists_path: str) -> None:
    """Save video lists to storage"""
    with open(video_lists_path, 'w', encoding='utf-8') as f:
        json.dump(lists, f, indent=2)

def load_video_summaries(video_summaries_path: str) -> Dict[str, Dict[str, Any]]:
    """Load video summaries from storage"""
    try:
        with open(video_summaries_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupt or missing, create a new empty one
        with open(video_summaries_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)
        return {}

def save_video_summaries(summaries: Dict[str, Dict[str, Any]], video_summaries_path: str) -> None:
    """Save video summaries to storage"""
    with open(video_summaries_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)

def load_all_videos_metadata(all_videos_metadata_path: str) -> Dict[str, Dict[str, Any]]:
    """Load all videos metadata from storage"""
    try:
        with open(all_videos_metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupt or missing, create a new empty one
        with open(all_videos_metadata_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)
        return {}

def save_all_videos_metadata(metadata: Dict[str, Dict[str, Any]], all_videos_metadata_path: str) -> None:
    """Save all videos metadata to storage"""
    with open(all_videos_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

def save_processed_data(processed_data: Dict[str, Any], output_file: str) -> str:
    """
    Save processed transcript data to a JSON file.
    
    Args:
        processed_data: Processed transcript data
        output_file: Path to output file
        
    Returns:
        Status message
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
        return f"Successfully saved processed data to {output_file}"
    except Exception as e:
        return f"Error saving data: {str(e)}"
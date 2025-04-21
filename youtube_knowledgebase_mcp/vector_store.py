"""
Vector Store Module

This module handles all vector store operations for the YouTube transcript knowledge base,
including creating, updating, and querying the FAISS index.
"""

import os
import time
import pickle
import shutil
from typing import List, Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Import alternative embedding models
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

def get_embeddings_model(embedding_type: str = None) -> Embeddings:
    """
    Get an appropriate embeddings model based on available API keys or local models.
    
    Args:
        embedding_type (str, optional): Explicitly choose an embedding model type.
            Options: 'openai', 'ollama', 'huggingface', or None (for automatic selection)
    
    Returns:
        An embeddings model instance that can be used with FAISS.
    """
    # If embedding type is explicitly specified, use that
    if embedding_type:
        embedding_type = embedding_type.lower()
        if embedding_type == 'openai':
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is required but not found in environment variables")
            return OpenAIEmbeddings(model="text-embedding-3-large")
        elif embedding_type == 'ollama':
            # Check if Ollama is running before trying to connect
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code != 200:
                    raise ValueError("Ollama server is not responding correctly. Make sure it's running with 'ollama serve'")
                return OllamaEmbeddings(model="llama2")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Cannot connect to Ollama server: {str(e)}. Make sure Ollama is installed and running with 'ollama serve'")
        elif embedding_type == 'huggingface':
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}. Choose from 'openai', 'ollama', or 'huggingface'")
    
    # Otherwise, automatic selection (try each option in order)
    errors = []
    
    # Try to use OpenAI embeddings if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddings(model="text-embedding-3-large")
        except Exception as e:
            errors.append(f"OpenAI embeddings error: {str(e)}")
            print(f"Error initializing OpenAI embeddings: {str(e)}")
    else:
        errors.append("No OpenAI API key found in environment variables")
    
    # Try Ollama (local model)
    try:
        print("Trying Ollama embeddings...")
        import requests
        # Check if Ollama is running before attempting to use it
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                return OllamaEmbeddings(model="llama2")
            else:
                errors.append(f"Ollama server responded with status code {response.status_code}")
                raise ValueError(f"Ollama server responded with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            errors.append(f"Cannot connect to Ollama server: {str(e)}")
            print(f"Cannot connect to Ollama server: {str(e)}. Is Ollama installed and running?")
            raise ValueError(f"Cannot connect to Ollama server: {str(e)}. Run 'ollama serve' to start the server.")
    except Exception as e:
        errors.append(f"Ollama embeddings error: {str(e)}")
        print(f"Error initializing Ollama embeddings: {str(e)}")
    
    # Fall back to HuggingFace embeddings as a last resort
    try:
        print("Falling back to HuggingFace embeddings...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        errors.append(f"HuggingFace embeddings error: {str(e)}")
        print(f"Error initializing HuggingFace embeddings: {str(e)}")
        
    # If all embedding methods failed, raise a detailed error
    error_message = "Could not initialize any embedding model. Errors encountered:\n"
    for i, err in enumerate(errors, 1):
        error_message += f"{i}. {err}\n"
    error_message += "\nPlease ensure one of the following:\n"
    error_message += "- Provide an OpenAI API key in the environment variables\n"
    error_message += "- Install and run Ollama with: curl -fsSL https://ollama.com/install.sh | sh && ollama serve\n"
    error_message += "- Ensure your internet connection is working for HuggingFace embeddings"
    
    raise ValueError(error_message)

def update_faiss_with_summary(video_id: str, summary: str, data_path: str, faiss_index_path: str, embedding_type: str = None) -> None:
    """
    Update FAISS index with video summary.
    
    Args:
        video_id: The YouTube video ID
        summary: Summary text to add to the index
        data_path: Path to data directory
        faiss_index_path: Path to FAISS index
        embedding_type: Optional embedding type to use ('openai', 'ollama', 'huggingface')
    """
    if not os.path.exists(faiss_index_path):
        return
    
    try:
        # Get metadata from file
        metadata_file = os.path.join(data_path, f"{video_id}_metadata.json")
        if not os.path.exists(metadata_file):
            return
            
        with open(metadata_file, 'r', encoding='utf-8') as f:
            import json
            metadata = json.load(f)
        
        # Create a document with the summary
        doc = Document(
            page_content=summary,
            metadata={
                "video_id": video_id,
                "content_type": "summary",
                "title": metadata.get("title", "Unknown"),
                "channel": metadata.get("channel", "Unknown"),
                "upload_date": metadata.get("upload_date", ""),
                "is_summary": True  # Flag to identify this as a summary
            }
        )
        
        # Add to FAISS index with specified embedding type
        vectorstore = get_or_create_faiss_index(faiss_index_path, [doc], embedding_type)
        print(f"Updated FAISS index with summary for video {video_id}")
    except Exception as e:
        print(f"Error updating FAISS with summary: {str(e)}")

def create_semantic_chunks(processed_data: Dict[str, Any], video_id: str, 
                           video_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
    """
    Create semantic chunks from a processed transcript and return as langchain Documents.
    
    Args:
        processed_data: Processed transcript data
        video_id: Video ID for reference
        video_metadata: Optional metadata about the video
        
    Returns:
        List of Document objects with metadata
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Check for valid transcript
    transcript = processed_data.get('transcript', '')
    if not transcript or transcript.startswith("No transcript") or transcript.startswith("Error"):
        # Create a single document with metadata indicating no content
        return [Document(
            page_content=f"No valid transcript content available for video {video_id}.",
            metadata={
                "video_id": video_id,
                "chunk_id": 0,
                "start_time": "00:00:00.000",
                "end_time": "00:00:10.000",
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "screenshots": [],
                "video_type": "unknown",
                "status": "empty_transcript",
                "title": video_metadata.get("title", "Unknown") if video_metadata else "Unknown",
                "channel": video_metadata.get("channel", "Unknown") if video_metadata else "Unknown",
                "upload_date": video_metadata.get("upload_date", "") if video_metadata else "",
                "description": video_metadata.get("description", "")[:500] if video_metadata else ""
            }
        )]
    
    # Use a text splitter to create semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the transcript into chunks
    chunks = text_splitter.split_text(transcript)
    documents = []
    
    # Get segments, ensuring we have at least an empty list
    segments = processed_data.get('segments', [])
    
    # Create a document for each chunk with metadata
    for i, chunk in enumerate(chunks):
        # Find timestamps that correspond to this chunk
        timestamp_range = find_timestamp_range(chunk, segments)
        
        # Find screenshots that correspond to this chunk
        screenshot_info = []
        for moment in processed_data.get('keyMoments', []):
            if moment['startSeconds'] >= timestamp_range['startSeconds'] and \
               moment['endSeconds'] <= timestamp_range['endSeconds']:
                screenshot_info.append({
                    'time': moment['startTime'],
                    'seconds': moment['startSeconds'],
                    'keywords': moment.get('keywords', []),
                    'categories': moment.get('categories', [])
                })
        
        # Create base metadata
        metadata_dict = {
            "video_id": video_id,
            "chunk_id": i,
            "start_time": timestamp_range['startTime'],
            "end_time": timestamp_range['endTime'],
            "start_seconds": timestamp_range['startSeconds'],
            "end_seconds": timestamp_range['endSeconds'],
            "screenshots": screenshot_info,
            "video_type": processed_data.get('metadata', {}).get('videoType', 'unknown')
        }
        
        # Add video metadata if available
        if video_metadata:
            metadata_dict.update({
                "title": video_metadata.get("title", "Unknown"),
                "channel": video_metadata.get("channel", "Unknown"),
                "upload_date": video_metadata.get("upload_date", ""),
                "description": video_metadata.get("description", "")[:500] if video_metadata.get("description") else "",
                "categories": video_metadata.get("categories", []),
                "tags": video_metadata.get("tags", []),
                "duration": video_metadata.get("duration", 0),
                "view_count": video_metadata.get("view_count", 0)
            })
        
        # Create a document with metadata
        doc = Document(
            page_content=chunk,
            metadata=metadata_dict
        )
        
        documents.append(doc)
    
    return documents

def find_timestamp_range(chunk: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find the timestamp range that corresponds to a chunk of text.
    
    Args:
        chunk: Text chunk
        segments: List of segments with timestamps
        
    Returns:
        Dictionary with start and end times
    """
    # Handle empty segments list
    if not segments:
        # Return default values if no segments are available
        return {
            'startTime': "00:00:00.000",
            'endTime': "00:00:10.000",
            'startSeconds': 0.0,
            'endSeconds': 10.0
        }
    
    # Simple approach: find segments that contain parts of the chunk
    matching_segments = []
    
    # Split chunk into smaller pieces to increase matching probability
    chunk_words = chunk.split()
    
    for segment in segments:
        segment_words = segment['content'].split()
        # Check for word overlap
        overlap = set(chunk_words).intersection(set(segment_words))
        if len(overlap) > 3:  # At least 3 words in common
            matching_segments.append(segment)
    
    if matching_segments:
        # Sort by start time
        matching_segments.sort(key=lambda x: x['startSeconds'])
        return {
            'startTime': matching_segments[0]['startTime'],
            'endTime': matching_segments[-1]['endTime'],
            'startSeconds': matching_segments[0]['startSeconds'],
            'endSeconds': matching_segments[-1]['endSeconds']
        }
    
    # Fallback if no matches found
    return {
        'startTime': segments[0]['startTime'],
        'endTime': segments[-1]['endTime'],
        'startSeconds': segments[0]['startSeconds'],
        'endSeconds': segments[-1]['endSeconds']
    }

def get_or_create_faiss_index(faiss_index_path: str, documents=None, embedding_type: str = None):
    """
    Get the existing FAISS index or create a new one.
    
    Args:
        faiss_index_path: Path to FAISS index directory
        documents: Optional list of documents to initialize with
        embedding_type: Optional embedding type to use ('openai', 'ollama', 'huggingface')
        
    Returns:
        FAISS vector store instance
    """
    embeddings = get_embeddings_model(embedding_type)
    
    # Check if FAISS index exists
    if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
        try:
            # Load existing FAISS index
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing FAISS index from {faiss_index_path}")
            
            if documents:
                # Add documents to existing index
                vectorstore.add_documents(documents)
                # Save the updated index
                vectorstore.save_local(faiss_index_path)
                print(f"Added {len(documents)} documents to FAISS index")
                
            return vectorstore
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            # If loading fails, create a new one
            if os.path.exists(faiss_index_path):
                # Backup the corrupted index
                backup_path = f"{faiss_index_path}_bak_{int(time.time())}"
                os.rename(faiss_index_path, backup_path)
                print(f"Backed up corrupted FAISS index to {backup_path}")
    
    # Create new FAISS index
    if documents:
        print(f"Creating new FAISS index with {len(documents)} documents")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(faiss_index_path)
        return vectorstore
    else:
        # Create empty FAISS index with a dummy document
        dummy_doc = Document(
            page_content="Initialization document for FAISS index",
            metadata={"video_id": "init", "type": "initialization"}
        )
        vectorstore = FAISS.from_documents([dummy_doc], embeddings)
        vectorstore.save_local(faiss_index_path)
        print("Created new FAISS index with initialization document")
        return vectorstore

def reset_knowledge_base(faiss_index_path: str) -> str:
    """
    Reset the knowledge base by deleting the FAISS index.
    
    Args:
        faiss_index_path: Path to FAISS index directory
        
    Returns:
        Status message
    """
    try:
        if os.path.exists(faiss_index_path):
            # Create a backup of the old index
            backup_path = f"{faiss_index_path}_bak_{int(time.time())}"
            shutil.copytree(faiss_index_path, backup_path)
            
            # Remove the old index
            shutil.rmtree(faiss_index_path)
            
            return f"Knowledge base reset. Old index backed up to {backup_path}"
        else:
            return "Knowledge base doesn't exist yet."
    except Exception as e:
        return f"Error resetting knowledge base: {str(e)}"
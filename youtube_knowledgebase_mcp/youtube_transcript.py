"""
YouTube Transcript Module

This module handles all YouTube transcript processing functionality, including:
- Extracting transcripts from YouTube videos
- Processing WebVTT content
- Extracting metadata from YouTube videos
"""

import os
import re
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse, parse_qs

# Import yt-dlp for direct YouTube integration
import yt_dlp

def extract_youtube_transcript(youtube_url: str) -> Tuple[str, str]:
    """
    Extract transcript from a YouTube video using the yt_dlp library.
    
    Args:
        youtube_url (str): YouTube video URL
        
    Returns:
        Tuple[str, str]: Video ID and WebVTT content
    """
    try:
        # Extract video ID from URL
        parsed_url = urlparse(youtube_url)
        if parsed_url.netloc == 'youtu.be':
            video_id = parsed_url.path[1:]
        elif 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            video_id = query_params.get('v', [''])[0]
        else:
            return "", f"Unsupported URL format: {youtube_url}"
        
        if not video_id:
            return "", f"Failed to extract video ID from URL: {youtube_url}"
        
        print(f"Extracted video ID: {video_id}")
        
        # Create a temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            # Configure yt-dlp options
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'en-US', 'en-GB'],
                'subtitlesformat': 'vtt',
                'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
                'quiet': False,  # Set to True to suppress console output
                'no_warnings': False,
            }
            
            # Create yt-dlp object
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info to trigger subtitle download
                try:
                    info = ydl.extract_info(youtube_url, download=True)
                    print(f"Video info extracted: {info.get('title', 'Unknown title')}")
                except yt_dlp.utils.DownloadError as e:
                    print(f"yt-dlp download error: {str(e)}")
                    # Try to continue anyway, as sometimes subtitles are downloaded
                    # despite errors in the main extraction
            
            # Look for subtitle files in the temp directory
            subtitle_files = [
                f for f in os.listdir(temp_dir) 
                if f.endswith('.vtt') and video_id in f
            ]
            
            if not subtitle_files:
                return video_id, "No subtitle files found after extraction."
            
            print(f"Found subtitle files: {subtitle_files}")
            
            # Prioritize manual subtitles over auto-generated ones
            # Sort order: 1. manual en, 2. manual en-US, 3. manual en-GB, 4. auto en, ...
            manual_subs = [f for f in subtitle_files if not '.auto.' in f]
            auto_subs = [f for f in subtitle_files if '.auto.' in f]
            
            if manual_subs:
                subtitle_file = os.path.join(temp_dir, manual_subs[0])
                print(f"Using manual subtitle: {manual_subs[0]}")
            elif auto_subs:
                subtitle_file = os.path.join(temp_dir, auto_subs[0])
                print(f"Using auto-generated subtitle: {auto_subs[0]}")
            else:
                return video_id, "No usable subtitle files found."
            
            # Read the subtitle file
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                webvtt_content = f.read()
            
            if not webvtt_content or len(webvtt_content) < 100:
                return video_id, f"Subtitle file is empty or too short: {len(webvtt_content)} bytes"
            
            return video_id, webvtt_content
                
    except Exception as e:
        import traceback
        error_message = f"Error extracting transcript: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return "", error_message

def process_webvtt_transcript(webvtt_content: str) -> Dict[str, Any]:
    """
    Process WebVTT content into a cleaned and structured format.
    
    Args:
        webvtt_content (str): Raw WebVTT content from YouTube
    
    Returns:
        dict: Structured transcript data with segments and key moments
    """
    # Validate WebVTT content
    if not webvtt_content or not webvtt_content.strip():
        # Generate an empty transcript with defaults
        return {
            "transcript": "No transcript content available.",
            "segments": [],
            "keyMoments": [],
            "metadata": {
                "videoType": "unknown",
                "processingNotes": "Empty or invalid WebVTT content"
            }
        }
    
    # Check for WEBVTT header
    if "WEBVTT" not in webvtt_content:
        # Not a valid WebVTT file, log a warning but continue processing
        print("Warning: Missing WEBVTT header in content. Attempting to process anyway.")
    
    # Try to parse the WebVTT content
    try:
        cues = parse_webvtt_content(webvtt_content)
    except Exception as e:
        # Parsing failed, return defaults with error info
        print(f"Error parsing WebVTT content: {str(e)}")
        return {
            "transcript": f"Error parsing WebVTT content: {str(e)}",
            "segments": [],
            "keyMoments": [],
            "metadata": {
                "videoType": "unknown",
                "processingNotes": f"WebVTT parsing error: {str(e)}"
            }
        }
    
    # Check if we got any cues
    if not cues:
        # No cues found, create a default segment to prevent errors
        default_segment = {
            'startTime': "00:00:00.000",
            'endTime': "00:00:10.000",
            'startSeconds': 0.0,
            'endSeconds': 10.0,
            'content': "No speech segments found in the transcript."
        }
        cues = [default_segment]
        clean_transcript = "No speech segments found in the transcript."
        key_moments = []
        video_type = "unknown"
    else:
        # Create a clean transcript
        clean_transcript = create_clean_transcript(cues)
        
        # Identify key moments for screenshots
        key_moments = identify_key_moments(cues)
        
        # Detect video type
        video_type = detect_video_type(clean_transcript)
    
    # Create the structured output
    return {
        "transcript": clean_transcript,
        "segments": cues,
        "keyMoments": key_moments,
        "metadata": {
            "videoType": video_type,
            "processingNotes": "Format optimized for summarization and screenshot capture"
        }
    }

def parse_webvtt_content(webvtt: str) -> List[Dict[str, Any]]:
    """
    Parse WebVTT content into structured data.
    
    Args:
        webvtt: Raw WebVTT content
        
    Returns:
        List of cue dictionaries with timing and text information
    """
    lines = webvtt.split('\n')
    cues = []
    
    current_cue = None
    
    # Skip the header
    i = 0
    while i < len(lines) and '-->' not in lines[i]:
        i += 1
    
    # Process each line
    for j in range(i, len(lines)):
        line = lines[j].strip()
        
        # Skip empty lines
        if line == '':
            continue
        
        # Process timestamp lines
        if '-->' in line:
            # If we have a current cue, add it to our collection
            if current_cue and current_cue['content'].strip():
                cues.append(current_cue)
            
            # Start a new cue
            time_parts = line.split('-->')
            start_time = time_parts[0].strip().split(' ')[0]
            end_time = time_parts[1].strip().split(' ')[0]
            
            current_cue = {
                'startTime': start_time,
                'endTime': end_time,
                'startSeconds': time_to_seconds(start_time),
                'endSeconds': time_to_seconds(end_time),
                'content': ''
            }
            
            continue
        
        # Skip alignment/positioning info
        if 'align:' in line:
            continue
        
        # Add content to current cue
        if current_cue:
            # Clean the line of HTML-like tags
            cleaned_line = re.sub(r'<[^>]*>', '', line).strip()
            
            if cleaned_line and '[Music]' not in cleaned_line and '[Applause]' not in cleaned_line:
                current_cue['content'] += (' ' if current_cue['content'] else '') + cleaned_line
    
    # Add the final cue if it exists
    if current_cue and current_cue['content'].strip():
        cues.append(current_cue)
    
    return merge_duplicate_cues(cues)

def time_to_seconds(time_str: str) -> float:
    """
    Convert time string to seconds.
    
    Args:
        time_str: Time in format HH:MM:SS.mmm
        
    Returns:
        Time in seconds
    """
    parts = time_str.split(':')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

def merge_duplicate_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge duplicate/overlapping cues.
    
    Args:
        cues: List of cue dictionaries
        
    Returns:
        Merged cues
    """
    if len(cues) <= 1:
        return cues
    
    merged_cues = [cues[0]]
    
    for i in range(1, len(cues)):
        current = cues[i]
        prev = merged_cues[-1]
        
        # Check if current cue is very similar to the previous one
        if calculate_similarity(current['content'], prev['content']) > 0.7:
            # Use the longer text
            if len(current['content']) > len(prev['content']):
                prev['content'] = current['content']
            
            # Update the end time if the current cue ends later
            if current['endSeconds'] > prev['endSeconds']:
                prev['endTime'] = current['endTime']
                prev['endSeconds'] = current['endSeconds']
        else:
            merged_cues.append(current)
    
    return merged_cues

def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    words1 = str1.lower().split()
    words2 = str2.lower().split()
    
    if not words1 or not words2:
        return 0
    
    common_words = [word for word in words1 if word in words2]
    return len(common_words) / max(len(words1), len(words2))

def create_clean_transcript(cues: List[Dict[str, Any]]) -> str:
    """
    Create a clean transcript from cue data, removing duplicates.
    
    Args:
        cues: List of cue dictionaries
        
    Returns:
        Clean transcript text
    """
    # Extract sentences from cues, removing duplicates
    sentences = []
    seen_phrases = set()
    
    for cue in cues:
        # Split the text into sentences/phrases
        phrases = [p.strip() for p in re.split(r'[.!?]', cue['content']) if p.strip()]
        
        for phrase in phrases:
            # Check if we've seen this phrase or a very similar one
            is_duplicate = False
            
            for existing in seen_phrases:
                if calculate_similarity(phrase, existing) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(phrase) > 0:
                sentences.append(phrase)
                seen_phrases.add(phrase)
    
    # Join sentences into a coherent transcript
    return '. '.join(sentences) + '.'

def identify_key_moments(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify potential key moments for screenshots.
    
    Args:
        cues: List of cue dictionaries
        
    Returns:
        List of key moments with metadata
    """
    # Keywords that might indicate visual content in different video types
    keywords_by_type = {
        'financial': ['chart', 'graph', 'market', 'stock', 'trend', 'volatility', 'price', 'investment'],
        'technical': ['diagram', 'architecture', 'code', 'implementation', 'demo', 'example'],
        'educational': ['figure', 'illustration', 'shown', 'demonstrate', 'example', 'observe'],
        'general': ['look', 'see', 'show', 'display', 'screen', 'visual', 'image', 'picture']
    }
    
    # Flatten keywords for initial search
    all_keywords = [kw for kws in keywords_by_type.values() for kw in kws]
    
    key_moments = []
    
    for cue in cues:
        lower_text = cue['content'].lower()
        matched_keywords = [keyword for keyword in all_keywords if keyword in lower_text]
        
        if matched_keywords:
            # Determine which category(s) of keywords matched
            categories = [
                category for category, keywords in keywords_by_type.items()
                if any(kw in matched_keywords for kw in keywords)
            ]
            
            key_moments.append({
                'startTime': cue['startTime'],
                'endTime': cue['endTime'],
                'startSeconds': cue['startSeconds'],
                'endSeconds': cue['endSeconds'],
                'content': cue['content'],
                'keywords': matched_keywords,
                'categories': categories
            })
    
    return key_moments

def detect_video_type(transcript: str) -> str:
    """
    Detect the likely type of video from transcript content.
    
    Args:
        transcript: Clean transcript text
        
    Returns:
        Detected video type
    """
    lower_text = transcript.lower()
    
    type_signals = {
        'financial': ['market', 'stock', 'invest', 'finance', 'economy', 'trading', 'portfolio'],
        'technical': ['code', 'programming', 'software', 'developer', 'algorithm', 'engineering'],
        'educational': ['learn', 'course', 'study', 'education', 'curriculum', 'teacher', 'student'],
        'entertainment': ['movie', 'show', 'music', 'game', 'play', 'fun', 'entertainment']
    }
    
    # Count occurrences of type signals
    type_counts = {}
    
    for type_name, signals in type_signals.items():
        type_counts[type_name] = sum(1 for signal in signals if signal in lower_text)
    
    # Find the type with the most signals (default to general if no clear match)
    if all(count == 0 for count in type_counts.values()):
        return 'general'
        
    dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
    return dominant_type

def load_webvtt_file(file_path: str) -> str:
    """
    Load WebVTT content from a file.
    
    Args:
        file_path: Path to WebVTT file
        
    Returns:
        WebVTT content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading file: {str(e)}"

def get_youtube_metadata(youtube_url: str) -> Dict[str, Any]:
    """
    Extract metadata from a YouTube video.
    
    Args:
        youtube_url: YouTube video URL
        
    Returns:
        Video metadata including title, description, etc.
    """
    try:
        # Configure yt-dlp options
        ydl_opts = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        # Create yt-dlp object
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info
            info = ydl.extract_info(youtube_url, download=False)
            
            # Return relevant metadata
            return {
                'video_id': info.get('id', ''),
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'upload_date': info.get('upload_date', ''),
                'channel': info.get('channel', ''),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'categories': info.get('categories', []),
                'tags': info.get('tags', []),
                'has_subtitles': info.get('requested_subtitles') is not None,
            }
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return {
            'error': str(e),
            'video_id': '',
            'title': 'Unknown',
        }
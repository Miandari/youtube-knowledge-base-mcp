"""
Context generation services for Contextual Retrieval.

This module implements Anthropic's Contextual Retrieval pattern to improve
RAG retrieval quality by generating contextual text for each chunk.
"""
from .provider import (
    ContextProvider,
    OpenAIContextProvider,
    OllamaContextProvider,
    get_context_provider,
)
from .contextualizer import (
    ContextualizationService,
    ContextualizedChunk,
    Chapter,
)

__all__ = [
    # Providers
    "ContextProvider",
    "OpenAIContextProvider",
    "OllamaContextProvider",
    "get_context_provider",
    # Service
    "ContextualizationService",
    "ContextualizedChunk",
    "Chapter",
]

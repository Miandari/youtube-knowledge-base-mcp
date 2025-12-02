"""
OpenAI-based context generation provider with prompt caching.

This module provides LLM-based context generation for Contextual Retrieval.
Uses sequential processing to leverage OpenAI's prompt caching for efficiency.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)


class ContextProvider(ABC):
    """Abstract base class for context generation providers."""

    CONTEXT_PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the whole document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    SUMMARY_PROMPT = """Summarize the following transcript in exactly {sentences} sentences. Focus on the main topic, key points, and unique insights.

Transcript:
{document}"""

    CHAPTER_SUMMARY_PROMPT = """This is chapter {chapter_num} of a longer video. Summarize this chapter in 2-3 sentences, focusing on the specific topics discussed in this segment.

Chapter content:
{chapter_content}"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @abstractmethod
    def generate_context(self, document: str, chunk: str, max_tokens: int = 100) -> str:
        """Generate context for a single chunk."""
        pass

    @abstractmethod
    def generate_summary(self, document: str, max_sentences: int = 3) -> str:
        """Generate a global summary of the document."""
        pass

    @abstractmethod
    def generate_chapter_summary(self, chapter_content: str, chapter_num: int) -> str:
        """Generate a summary for a specific chapter."""
        pass


class OpenAIContextProvider(ContextProvider):
    """
    OpenAI gpt-4o-mini context provider.

    Uses SEQUENTIAL processing to leverage OpenAI's prompt caching:
    - First request with transcript: full cost
    - Subsequent requests: 50% discount, ~5x faster (cached prefix)

    This is faster than parallel processing for same-document chunks due to caching.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI context provider.

        Args:
            model: The OpenAI model to use (default: gpt-4o-mini)

        Raises:
            ValueError: If OPENAI_API_KEY is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def generate_context(self, document: str, chunk: str, max_tokens: int = 100) -> str:
        """
        Generate context for a single chunk.

        Note: Called sequentially to leverage prompt caching on the document prefix.

        Args:
            document: The full document/transcript text
            chunk: The specific chunk to generate context for
            max_tokens: Maximum tokens for the context response

        Returns:
            Generated context string, or empty string on failure
        """
        prompt = self.CONTEXT_PROMPT.format(document=document, chunk=chunk)
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Context generation failed for chunk: {e}")
            return ""

    def generate_summary(self, document: str, max_sentences: int = 3) -> str:
        """
        Generate a global summary of the document.

        Args:
            document: The document text to summarize
            max_sentences: Number of sentences for the summary

        Returns:
            Generated summary string, or empty string on failure
        """
        prompt = self.SUMMARY_PROMPT.format(document=document, sentences=max_sentences)
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return ""

    def generate_chapter_summary(self, chapter_content: str, chapter_num: int) -> str:
        """
        Generate a summary for a specific chapter of a long video.

        Args:
            chapter_content: The chapter's text content
            chapter_num: The chapter number (1-indexed for display)

        Returns:
            Generated chapter summary, or empty string on failure
        """
        prompt = self.CHAPTER_SUMMARY_PROMPT.format(
            chapter_content=chapter_content,
            chapter_num=chapter_num,
        )
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Chapter summary generation failed for chapter {chapter_num}: {e}")
            return ""


class OllamaContextProvider(ContextProvider):
    """
    Ollama-based context provider for local LLM inference.

    Useful for privacy-sensitive deployments or when API costs are a concern.
    """

    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize the Ollama context provider.

        Args:
            model: The Ollama model to use

        Raises:
            ValueError: If Ollama is not available or model not found
        """
        import httpx
        self._model = model
        self._base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = httpx.Client(timeout=120.0)

        # Verify Ollama is available
        try:
            response = self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
        except Exception as e:
            raise ValueError(f"Could not connect to Ollama at {self._base_url}: {e}")

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return f"ollama:{self._model}"

    def generate_context(self, document: str, chunk: str, max_tokens: int = 100) -> str:
        """Generate context using Ollama."""
        prompt = self.CONTEXT_PROMPT.format(document=document, chunk=chunk)
        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0}
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            logger.warning(f"Context generation failed for chunk: {e}")
            return ""

    def generate_summary(self, document: str, max_sentences: int = 3) -> str:
        """Generate summary using Ollama."""
        prompt = self.SUMMARY_PROMPT.format(document=document, sentences=max_sentences)
        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 200, "temperature": 0}
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return ""

    def generate_chapter_summary(self, chapter_content: str, chapter_num: int) -> str:
        """Generate chapter summary using Ollama."""
        prompt = self.CHAPTER_SUMMARY_PROMPT.format(
            chapter_content=chapter_content,
            chapter_num=chapter_num,
        )
        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 150, "temperature": 0}
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            logger.warning(f"Chapter summary generation failed for chapter {chapter_num}: {e}")
            return ""


def get_context_provider(provider: str = "openai", model: Optional[str] = None) -> ContextProvider:
    """
    Factory function to get a context provider.

    Args:
        provider: The provider type ("openai" or "ollama")
        model: Optional model name override

    Returns:
        Configured ContextProvider instance

    Raises:
        ValueError: If provider type is unknown
    """
    if provider == "openai":
        return OpenAIContextProvider(model=model or "gpt-4o-mini")
    elif provider == "ollama":
        return OllamaContextProvider(model=model or "llama3.2:3b")
    else:
        raise ValueError(f"Unknown context provider: {provider}")

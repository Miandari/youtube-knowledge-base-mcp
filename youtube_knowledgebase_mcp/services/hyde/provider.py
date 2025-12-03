"""
HyDE (Hypothetical Document Embeddings) providers.

Generate hypothetical transcript snippets that would answer the query,
bridging vocabulary gaps between formal search terms and informal transcripts.
"""
from abc import ABC, abstractmethod
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class HyDEProvider(ABC):
    """Abstract base class for HyDE providers."""

    HYDE_PROMPT = """You are helping improve search over YouTube video transcripts.

Given a search query, generate a SHORT hypothetical transcript snippet (2-3 sentences)
that a YouTube creator might say when discussing this topic.

Use informal, conversational language typical of video content. Include:
- Common synonyms and related terms
- How someone might naturally explain this concept
- Any technical jargon AND its casual alternatives

Query: {query}

Output ONLY the transcript snippet. No preamble, no "Here is...", no quotation marks. Start directly with what the speaker would say."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def generate_hypothetical_document(self, query: str, max_tokens: int = 150) -> str:
        """Generate a hypothetical document for the query."""
        pass


class OpenAIHyDEProvider(HyDEProvider):
    """OpenAI-based HyDE provider."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the OpenAI HyDE provider.

        Args:
            model: The OpenAI model to use
            temperature: Sampling temperature for generation

        Raises:
            ValueError: If OPENAI_API_KEY is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def generate_hypothetical_document(self, query: str, max_tokens: int = 150) -> str:
        """
        Generate a hypothetical document for the query.

        Args:
            query: The search query
            max_tokens: Maximum tokens for the response

        Returns:
            Hypothetical document text, or original query on failure
        """
        prompt = self.HYDE_PROMPT.format(query=query)
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query  # Fallback to original


class OllamaHyDEProvider(HyDEProvider):
    """Ollama-based HyDE provider for local inference."""

    def __init__(self, model: str = "llama3.2:3b", temperature: float = 0.7):
        """
        Initialize the Ollama HyDE provider.

        Args:
            model: The Ollama model to use
            temperature: Sampling temperature for generation
        """
        import httpx
        self._model = model
        self._temperature = temperature
        self._base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = httpx.Client(timeout=60.0)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return f"ollama:{self._model}"

    def generate_hypothetical_document(self, query: str, max_tokens: int = 150) -> str:
        """
        Generate a hypothetical document for the query.

        Args:
            query: The search query
            max_tokens: Maximum tokens for the response

        Returns:
            Hypothetical document text, or original query on failure
        """
        prompt = self.HYDE_PROMPT.format(query=query)
        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": self._temperature
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query  # Fallback to original


def get_hyde_provider(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> HyDEProvider:
    """
    Factory function to get a HyDE provider.

    Args:
        provider: The provider type ("openai" or "ollama")
        model: Optional model name override
        temperature: Sampling temperature for generation

    Returns:
        Configured HyDEProvider instance

    Raises:
        ValueError: If provider type is unknown
    """
    if provider == "openai":
        return OpenAIHyDEProvider(model=model or "gpt-4o-mini", temperature=temperature)
    elif provider == "ollama":
        return OllamaHyDEProvider(model=model or "llama3.2:3b", temperature=temperature)
    else:
        raise ValueError(f"Unknown HyDE provider: {provider}")

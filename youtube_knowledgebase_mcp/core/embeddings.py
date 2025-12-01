"""
Embedding provider abstraction.
Supports multiple embedding backends with automatic fallback.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import os


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the full model identifier."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass


class VoyageEmbedding(EmbeddingProvider):
    """Voyage AI embedding provider."""

    def __init__(self, model: str = "voyage-3-large", dimensions: int = 1024):
        import voyageai
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY environment variable not set")
        self.client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return f"voyage:{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.client.embed(
            texts,
            model=self._model,
            input_type="document",
            truncation=True
        )
        return result.embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.embed(
            [text],
            model=self._model,
            input_type="query",
            truncation=True
        )
        return result.embeddings[0]


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, model: str = "text-embedding-3-large", dimensions: int = 1024):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return f"openai:{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self._model,
            input=[text],
            dimensions=self._dimensions
        )
        return response.data[0].embedding


class BGEEmbedding(EmbeddingProvider):
    """BGE (HuggingFace) embedding provider - runs locally."""

    def __init__(self, model: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model
        self.model = SentenceTransformer(model)
        self._dimensions = self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return f"bge:{self._model_name}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


class OllamaEmbedding(EmbeddingProvider):
    """Ollama embedding provider - runs locally via REST API."""

    def __init__(self, model: str = "nomic-embed-text", dimensions: int = 768):
        import httpx
        self._model = model
        self._dimensions = dimensions
        self._base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = httpx.Client(timeout=60.0)

        # Verify Ollama is running
        try:
            response = self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
        except Exception as e:
            raise ValueError(f"Could not connect to Ollama at {self._base_url}: {e}")

    @property
    def model_name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text using Ollama API."""
        response = self._client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ollama doesn't support batch embedding, so we iterate
        return [self._embed_single(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)


def get_embedding_provider(
    provider: str = "voyage",
    model: Optional[str] = None,
    dimensions: int = 1024
) -> EmbeddingProvider:
    """
    Factory function to get an embedding provider.
    Falls back through providers if preferred one is unavailable.

    Fallback order: voyage -> openai -> bge -> ollama

    Note: When falling back, the model parameter is ignored and the default
    model for the fallback provider is used.
    """
    # Default models for each provider
    default_models = {
        "voyage": "voyage-3-large",
        "openai": "text-embedding-3-large",
        "bge": "BAAI/bge-m3",
        "ollama": "nomic-embed-text",
    }

    providers_to_try = []

    # Build ordered list based on preference
    if provider == "voyage":
        providers_to_try = ["voyage", "openai", "bge", "ollama"]
    elif provider == "openai":
        providers_to_try = ["openai", "voyage", "bge", "ollama"]
    elif provider == "bge":
        providers_to_try = ["bge", "voyage", "openai", "ollama"]
    elif provider == "ollama":
        providers_to_try = ["ollama", "bge", "voyage", "openai"]
    else:
        providers_to_try = [provider, "voyage", "openai", "bge", "ollama"]

    errors = []

    for i, p in enumerate(providers_to_try):
        try:
            # Use provided model only for the preferred provider, default for fallbacks
            use_model = model if (i == 0 and model) else default_models.get(p)

            if p == "voyage":
                return VoyageEmbedding(model=use_model or "voyage-3-large", dimensions=dimensions)
            elif p == "openai":
                return OpenAIEmbedding(model=use_model or "text-embedding-3-large", dimensions=dimensions)
            elif p == "bge":
                return BGEEmbedding(model=use_model or "BAAI/bge-m3")
            elif p == "ollama":
                return OllamaEmbedding(model=use_model or "nomic-embed-text", dimensions=dimensions)
        except Exception as e:
            errors.append(f"{p}: {str(e)}")
            continue

    raise RuntimeError(f"Could not initialize any embedding provider. Errors: {errors}")

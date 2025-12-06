"""
Embedding provider abstraction.
Supports multiple embedding backends with automatic fallback.
"""
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # voyage-3-large is natively 1024 dimensions
        if dimensions != 1024:
            raise ValueError(f"voyage-3-large requires dimensions=1024, got {dimensions}")
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

    def __init__(self, model: str = "BAAI/bge-m3", dimensions: int = 1024):
        from sentence_transformers import SentenceTransformer
        self._model_name = model
        self.model = SentenceTransformer(model)
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != dimensions:
            raise ValueError(f"Model {model} has {actual_dim} dimensions, expected {dimensions}")
        self._dimensions = actual_dim

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


class LocalChunkingEmbedding(EmbeddingProvider):
    """
    Lightweight local embedding for semantic chunking only.

    Uses all-MiniLM-L6-v2 (~80MB) for topic shift detection.
    Much cheaper than API calls for determining chunk boundaries.
    NOT suitable for final chunk embeddings - use Voyage/OpenAI for that.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local chunking. "
                "Install with: pip install sentence-transformers"
            )
        self._model_name = model
        self.model = SentenceTransformer(model)
        self._dimensions = self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return f"local:{self._model_name}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


def get_chunking_embedding_provider() -> Optional[EmbeddingProvider]:
    """
    Get a local embedding provider for semantic chunking.

    Returns None if sentence-transformers is not installed,
    allowing graceful fallback to sentence-boundary chunking.
    """
    try:
        return LocalChunkingEmbedding()
    except ImportError:
        return None


class OllamaEmbedding(EmbeddingProvider):
    """Ollama embedding provider - runs locally via REST API."""

    def __init__(self, model: str = "mxbai-embed-large", dimensions: int = 1024):
        import httpx
        self._httpx = httpx  # Store for exception handling
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

        # Probe model dimensions with a test embedding
        try:
            test_response = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": "test"}
            )
            test_response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self._model}' not found. "
                    f"Please run: ollama pull {self._model}"
                ) from e
            raise

        actual_dim = len(test_response.json()["embedding"])
        if actual_dim != dimensions:
            raise ValueError(
                f"Model {self._model} produces {actual_dim} dimensions, expected {dimensions}"
            )

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

    def embed_documents(self, texts: List[str], max_workers: int = 5) -> List[List[float]]:
        """Embed documents in parallel using thread pool."""
        if not texts:
            return []

        results = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._embed_single, text): i
                for i, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)


def get_embedding_provider(
    provider: str = "voyage",
    model: Optional[str] = None,
    dimensions: int = 1024
) -> EmbeddingProvider:
    """
    Factory function to get an embedding provider.

    NO FALLBACK - fails immediately with helpful error if provider unavailable.
    This prevents silent data corruption from mixing embeddings from different models.

    Args:
        provider: The embedding provider to use (voyage, openai, bge, ollama)
        model: Optional model name override
        dimensions: Vector dimensions (default 1024)

    Returns:
        Configured EmbeddingProvider

    Raises:
        ConfigurationError: If the provider cannot be initialized (e.g., missing API key)
    """
    from .config import ConfigurationError

    # Default models for each provider
    default_models = {
        "voyage": "voyage-3-large",
        "openai": "text-embedding-3-large",
        "bge": "BAAI/bge-m3",
        "ollama": "mxbai-embed-large",
    }

    use_model = model or default_models.get(provider, "")

    try:
        if provider == "voyage":
            if not os.getenv("VOYAGE_API_KEY"):
                raise ConfigurationError(
                    "VOYAGE_API_KEY not set.\n"
                    "Either:\n"
                    "  1. Set VOYAGE_API_KEY in your environment\n"
                    "  2. Change EMBEDDING_PROVIDER to 'openai', 'bge', or 'ollama'"
                )
            return VoyageEmbedding(model=use_model, dimensions=dimensions)

        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ConfigurationError(
                    "OPENAI_API_KEY not set.\n"
                    "Either:\n"
                    "  1. Set OPENAI_API_KEY in your environment\n"
                    "  2. Change EMBEDDING_PROVIDER to 'voyage', 'bge', or 'ollama'"
                )
            return OpenAIEmbedding(model=use_model, dimensions=dimensions)

        elif provider == "bge":
            try:
                return BGEEmbedding(model=use_model, dimensions=dimensions)
            except ImportError:
                raise ConfigurationError(
                    "sentence-transformers not installed for BGE embeddings.\n"
                    "Either:\n"
                    "  1. Install: pip install sentence-transformers\n"
                    "  2. Change EMBEDDING_PROVIDER to 'voyage' or 'openai'"
                )

        elif provider == "ollama":
            return OllamaEmbedding(model=use_model, dimensions=dimensions)

        else:
            raise ConfigurationError(
                f"Unknown embedding provider: '{provider}'.\n"
                f"Valid options: voyage, openai, bge, ollama"
            )

    except ConfigurationError:
        raise  # Re-raise our custom errors
    except Exception as e:
        raise ConfigurationError(
            f"Failed to initialize {provider} embedding provider: {e}\n"
            f"Check your configuration and try again."
        )

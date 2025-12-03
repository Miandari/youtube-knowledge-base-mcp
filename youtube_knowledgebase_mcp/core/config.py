"""
Configuration management for the knowledge base.
"""
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    provider: Literal["voyage", "openai", "bge", "ollama"] = "voyage"
    model: Optional[str] = None  # Auto-selected based on provider if None
    dimensions: int = 1024

    def get_model_name(self) -> str:
        """Return the actual model name for the provider."""
        if self.model:
            return self.model
        defaults = {
            "voyage": "voyage-3-large",
            "openai": "text-embedding-3-large",
            "bge": "BAAI/bge-m3",
            "ollama": "mxbai-embed-large",
        }
        return defaults[self.provider]


class ContextConfig(BaseModel):
    """Contextual retrieval configuration."""
    enabled: bool = True
    provider: Literal["openai", "ollama"] = "openai"
    model: Optional[str] = None  # gpt-4o-mini for openai
    max_context_tokens: int = 100
    include_summary: bool = True
    summary_sentences: int = 3
    chapter_duration_minutes: int = 30  # For long video chaptering
    max_transcript_tokens: int = 100000  # Trigger chaptering above this

    def get_model_name(self) -> str:
        """Return the actual model name for the provider."""
        if self.model:
            return self.model
        return {"openai": "gpt-4o-mini", "ollama": "llama3.2:3b"}[self.provider]


class RerankConfig(BaseModel):
    """Cross-encoder reranking configuration."""
    enabled: bool = True
    model_name: str = "ms-marco-MiniLM-L-12-v2"  # Best precision (~34MB)
    max_length: int = 256  # query + chunk tokens
    candidate_multiplier: int = 5  # Fetch 5x candidates for reranking
    cache_dir: str = "./opt/reranker"  # Local model cache


class HyDEConfig(BaseModel):
    """HyDE (Hypothetical Document Embeddings) configuration."""
    enabled: bool = True  # Bridges formal↔informal vocabulary gap
    provider: Literal["openai", "ollama"] = "openai"
    model: Optional[str] = None
    max_tokens: int = 150
    temperature: float = 0.7

    def get_model_name(self) -> str:
        """Return the actual model name for the provider."""
        if self.model:
            return self.model
        return {"openai": "gpt-4o-mini", "ollama": "llama3.2:3b"}[self.provider]


class Settings(BaseModel):
    """Application settings."""
    # Paths
    base_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")

    # Database
    db_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "knowledge.lance")

    # Embedding
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # Search
    default_search_limit: int = 10

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 150

    # Contextual Retrieval
    context: ContextConfig = Field(default_factory=ContextConfig)

    # Reranking
    rerank: RerankConfig = Field(default_factory=RerankConfig)

    # HyDE (Hypothetical Document Embeddings)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)

    # API Keys (from environment)
    voyage_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("VOYAGE_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    model_config = {"arbitrary_types_allowed": True}


# Global settings instance
settings = Settings()

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
            "ollama": "nomic-embed-text",
        }
        return defaults[self.provider]


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
    recency_boost_enabled: bool = True
    recency_floor: float = 0.5  # Minimum weight for old content

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 150

    # API Keys (from environment)
    voyage_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("VOYAGE_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    model_config = {"arbitrary_types_allowed": True}


# Global settings instance
settings = Settings()

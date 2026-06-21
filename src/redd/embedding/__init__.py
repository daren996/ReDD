"""Embedding and embedding-driven preprocessing utilities."""

from __future__ import annotations

from .base import EmbeddingProviderBase
from .clustering import (
    Clusterer,
    ClustererGPT,
    ClustererKMeans,
    DocumentClustering,
    DocVectorizer,
    Vectorizer,
    llm_embeddings,
)
from .config import (
    DEFAULT_EMBEDDING_STORAGE_FILE,
    embedding_config_value,
    embedding_manager_kwargs,
    resolve_embedding_storage_path,
)
from .manager import EmbeddingManager
from .providers import (
    PROVIDER_CONFIGS,
    EmbeddingProvider,
    detect_provider,
    get_embedding_provider,
)

__all__ = [
    "Clusterer",
    "ClustererGPT",
    "ClustererKMeans",
    "DocVectorizer",
    "DocumentClustering",
    "EmbeddingManager",
    "EmbeddingProvider",
    "EmbeddingProviderBase",
    "DEFAULT_EMBEDDING_STORAGE_FILE",
    "PROVIDER_CONFIGS",
    "Vectorizer",
    "detect_provider",
    "embedding_config_value",
    "embedding_manager_kwargs",
    "get_embedding_provider",
    "llm_embeddings",
    "resolve_embedding_storage_path",
]

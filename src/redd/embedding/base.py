from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EmbeddingProviderBase(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, model: str, api_key: str | None = None, **_: Any) -> None:
        self.model = model
        self.api_key = api_key
        self.embedding_dim: int | None = None

    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for one text input."""

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            for text in texts[start : start + batch_size]:
                embeddings.append(self.embed_single(text))
        return embeddings

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.__class__.__name__,
            "embedding_dim": self.embedding_dim,
        }

    @staticmethod
    def cosine_similarity(emb1: list[float], emb2: list[float]) -> float:
        arr1 = np.array(emb1, dtype=np.float32)
        arr2 = np.array(emb2, dtype=np.float32)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(arr1, arr2) / (norm1 * norm2))

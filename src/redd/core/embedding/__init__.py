"""Internal embedding components for ReDD.

Public callers should import from `redd.embedding`.
"""

from .base import EmbeddingProviderBase
from .manager import EmbeddingManager
from .providers import (
    PROVIDER_CONFIGS,
    EmbeddingProvider,
    detect_provider,
    get_embedding_provider,
)

__all__ = [
    "EmbeddingProviderBase",
    "EmbeddingManager",
    "EmbeddingProvider",
    "PROVIDER_CONFIGS",
    "detect_provider",
    "get_embedding_provider",
]

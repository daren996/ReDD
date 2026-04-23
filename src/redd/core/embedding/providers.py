from __future__ import annotations

import os
from typing import Any

from .base import EmbeddingProviderBase


PROVIDER_CONFIGS: dict[str, dict[str, str | None]] = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.com/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
    },
}

MODEL_PROVIDER_PATTERNS: list[tuple[str, str]] = [
    ("text-embedding-", "openai"),
    ("ada-", "openai"),
    ("deepseek-", "deepseek"),
    ("Qwen/Qwen3-Embedding", "siliconflow"),
    ("togethercomputer/", "together"),
    ("gemini-embedding-", "gemini"),
]


def detect_provider(model: str) -> str:
    for prefix, provider in MODEL_PROVIDER_PATTERNS:
        if model.startswith(prefix):
            return provider
    return "openai"


class EmbeddingProvider(EmbeddingProviderBase):
    """OpenAI-compatible embedding provider for multiple backends."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)

        self.provider_name = (provider or detect_provider(model)).lower()
        provider_config = PROVIDER_CONFIGS.get(self.provider_name, {})
        self.base_url = base_url or provider_config.get("base_url")

        resolved_api_key = api_key
        if resolved_api_key is None:
            env_name = provider_config.get("api_key_env")
            if env_name:
                resolved_api_key = os.getenv(env_name)
        if resolved_api_key is None:
            raise ValueError(
                f"API key required for embedding provider `{self.provider_name}`. "
                "Provide `api_key=` or the provider environment variable."
            )

        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)
        self.api_key = resolved_api_key

    def embed_single(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
        )
        embedding = response.data[0].embedding
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            batch_data = sorted(response.data, key=lambda item: item.index)
            batch_embeddings = [item.embedding for item in batch_data]
            if self.embedding_dim is None and batch_embeddings:
                self.embedding_dim = len(batch_embeddings[0])
            embeddings.extend(batch_embeddings)
        return embeddings


def get_embedding_provider(
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    return EmbeddingProvider(
        model=model,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
        **kwargs,
    )

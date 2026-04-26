from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from redd.exceptions import RuntimeDependencyError

from .base import EmbeddingProviderBase

PROVIDER_CONFIGS: dict[str, dict[str, str | None]] = {
    "openai": {
        "litellm_provider": "openai",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "litellm_provider": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "siliconflow": {
        "litellm_provider": "openai",
        "base_url": "https://api.siliconflow.com/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
    },
    "together": {
        "litellm_provider": "together_ai",
        "base_url": None,
        "api_key_env": "TOGETHER_API_KEY",
    },
    "gemini": {
        "litellm_provider": "gemini",
        "base_url": None,
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
    ("local-hash", "local"),
]


def detect_provider(model: str) -> str:
    for prefix, provider in MODEL_PROVIDER_PATTERNS:
        if model.startswith(prefix):
            return provider
    return "openai"


def _litellm_model_name(provider_config: dict[str, str | None], model: str) -> str:
    known_prefixes = (
        "openai/",
        "deepseek/",
        "together_ai/",
        "gemini/",
        "anthropic/",
        "azure/",
        "openrouter/",
    )
    if model.startswith(known_prefixes):
        return model
    provider = provider_config.get("litellm_provider")
    if not provider:
        return model
    return f"{provider}/{model}"


def _embedding_values(response: Any) -> list[list[float]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if data is None:
        return []

    def _index(item: Any) -> int:
        if isinstance(item, dict):
            return int(item.get("index", 0))
        return int(getattr(item, "index", 0))

    def _embedding(item: Any) -> list[float]:
        if isinstance(item, dict):
            return list(item["embedding"])
        return list(item.embedding)

    return [_embedding(item) for item in sorted(data, key=_index)]


class EmbeddingProvider(EmbeddingProviderBase):
    """LiteLLM-backed embedding provider for multiple backends."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        wait_time: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)

        self.provider_name = (provider or detect_provider(model)).lower()
        if self.provider_name in {"local", "none"}:
            self.embedding_dim = int(kwargs.get("embedding_dim", 256))
            self.api_key = None
            self.base_url = None
            self.litellm_model = model
            self.max_retries = int(max_retries)
            self.wait_time = float(wait_time)
            return
        provider_config = PROVIDER_CONFIGS.get(self.provider_name, PROVIDER_CONFIGS["openai"])
        self.base_url = base_url or provider_config.get("base_url")
        self.litellm_model = _litellm_model_name(provider_config, model)
        self.max_retries = int(max_retries)
        self.wait_time = float(wait_time)

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
        self.api_key = resolved_api_key

    def embed_single(self, text: str) -> list[float]:
        embeddings = self.embed_batch([text], batch_size=1)
        return embeddings[0]

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        if self.provider_name in {"local", "none"}:
            return [self._embed_local_hash(text) for text in texts]

        try:
            from litellm import embedding
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError("Embedding calls require `litellm`.") from exc

        embeddings: list[list[float]] = []

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=self.wait_time, min=self.wait_time),
            reraise=True,
        )
        def _call(batch: list[str]) -> Any:
            kwargs: dict[str, Any] = {
                "model": self.litellm_model,
                "input": batch,
                "api_key": self.api_key,
            }
            if self.base_url:
                kwargs["api_base"] = self.base_url
            return embedding(**kwargs)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_embeddings = _embedding_values(_call(batch))
            if self.embedding_dim is None and batch_embeddings:
                self.embedding_dim = len(batch_embeddings[0])
            embeddings.extend(batch_embeddings)
        return embeddings

    def _embed_local_hash(self, text: str) -> list[float]:
        dim = int(self.embedding_dim or 256)
        vector = [0.0] * dim
        tokens = re.findall(r"[a-z0-9_]+", str(text).lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % dim
            sign = 1.0 if (raw >> 8) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


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

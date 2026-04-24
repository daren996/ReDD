from __future__ import annotations

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from redd.exceptions import RuntimeDependencyError


@dataclass(frozen=True)
class ProviderSpec:
    """Runtime information for a model provider."""

    canonical_name: str
    api_key_env: str | None = None
    base_url: str | None = None
    is_local: bool = False


_PROVIDER_REGISTRY: dict[str, ProviderSpec] = {}
_PROVIDER_ALIASES: dict[str, str] = {}


class _LocalChatCompletions:
    def __init__(self, client: "LocalChatClient") -> None:
        self._client = client

    def create(self, **request_kwargs: Any) -> Any:
        return self._client.create_completion(**request_kwargs)


class LocalChatClient:
    """Minimal OpenAI-compatible chat client wrapper for local transformer models."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError(
                "Local provider support requires `torch` and `transformers`."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeDependencyError(
                "Local provider support currently requires CUDA."
            )

        model_name = str(config.get("llm_model") or "").strip()
        model_path = str(config.get("llm_model_path") or "").strip()
        if not model_name and not model_path:
            raise ValueError(
                "Local provider requires `llm_model` or `llm_model_path` in config."
            )

        model_ref = model_path or model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda()
        self._torch = torch
        self.chat = SimpleNamespace(completions=_LocalChatCompletions(self))

    def create_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        **_: Any,
    ) -> Any:
        del model
        input_tensor = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        with self._torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_tensor,
                max_new_tokens=int(max_tokens or 1000),
            )
        generated_tokens = outputs[0][input_tensor.shape[1] :]
        content = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


def register_provider(
    name: str,
    *,
    api_key_env: str | None = None,
    base_url: str | None = None,
    aliases: tuple[str, ...] = (),
    is_local: bool = False,
) -> None:
    canonical = name.strip().lower()
    spec = ProviderSpec(
        canonical_name=canonical,
        api_key_env=api_key_env,
        base_url=base_url,
        is_local=is_local,
    )
    _PROVIDER_REGISTRY[canonical] = spec
    _PROVIDER_ALIASES[canonical] = canonical
    for alias in aliases:
        _PROVIDER_ALIASES[alias.strip().lower()] = canonical


def normalize_provider_name(provider: str) -> str:
    normalized = str(provider).strip().lower()
    try:
        return _PROVIDER_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(f"Unsupported provider `{provider}`. Supported providers: {supported}") from exc


def get_provider_spec(provider: str) -> ProviderSpec:
    return _PROVIDER_REGISTRY[normalize_provider_name(provider)]


def is_local_provider(provider: str) -> bool:
    return get_provider_spec(provider).is_local


def get_api_key(
    config: Mapping[str, Any] | None,
    mode: str,
    api_key: str | None = None,
) -> str | None:
    """Resolve API key with priority: explicit arg > config['api_key'] > environment."""
    spec = get_provider_spec(mode)
    if spec.is_local:
        return None

    if api_key:
        return api_key

    if config and config.get("api_key"):
        return str(config["api_key"])

    if spec.api_key_env:
        env_api_key = os.getenv(spec.api_key_env)
        if env_api_key:
            return env_api_key

    raise ValueError(
        f"API key is required for provider `{spec.canonical_name}`. "
        "Provide it via `api_key=`, config['api_key'], or the provider environment variable."
    )


def create_client(
    mode: str,
    *,
    config: Mapping[str, Any] | None = None,
    api_key: str | None = None,
) -> Any | None:
    spec = get_provider_spec(mode)
    if spec.is_local:
        if config is None:
            raise ValueError("Local provider client creation requires a config mapping.")
        return LocalChatClient(config)

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "The `openai` package is required for cloud-backed provider clients. "
            "Install project dependencies first."
        ) from exc

    resolved_api_key = get_api_key(config, spec.canonical_name, api_key)
    client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    if spec.base_url:
        client_kwargs["base_url"] = spec.base_url
    return OpenAI(**client_kwargs)


def llm_completion(
    mode: str,
    client: Any | None,
    messages: list[dict[str, Any]],
    model: str,
    **kwargs: Any,
) -> str:
    """Execute a chat completion with shared retry logic for all cloud providers."""
    spec = get_provider_spec(mode)
    if spec.is_local:
        if client is None:
            raise ValueError("LLM client is required for local providers.")
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens"),
        )
        return completion.choices[0].message.content or ""

    try:
        from openai import APIError, RateLimitError
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "The `openai` package is required for cloud-backed LLM completion."
        ) from exc

    if client is None:
        raise ValueError("LLM client is required for cloud providers.")

    max_retries = int(kwargs.get("max_retries", 5))
    initial_wait = float(kwargs.get("wait_time", 10.0))
    response_format = kwargs.get("response_format", "json_object")

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if response_format:
        request_kwargs["response_format"] = {"type": response_format}

    for key in ("temperature", "top_p", "max_tokens", "seed"):
        value = kwargs.get(key)
        if value is not None:
            request_kwargs[key] = value

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(**request_kwargs)
            return completion.choices[0].message.content or ""
        except RateLimitError:
            if attempt >= max_retries:
                raise
            time.sleep(initial_wait * (2 ** attempt))
        except APIError:
            if attempt >= max_retries:
                raise
            time.sleep(initial_wait * (2 ** attempt))

    raise RuntimeError("LLM completion failed unexpectedly.")


register_provider("cgpt", api_key_env="OPENAI_API_KEY", aliases=("openai", "gpt"))
register_provider(
    "deepseek",
    api_key_env="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com",
)
register_provider(
    "together",
    api_key_env="TOGETHER_API_KEY",
    base_url="https://api.together.ai/v1",
)
register_provider(
    "siliconflow",
    api_key_env="SILICONFLOW_API_KEY",
    base_url="https://api.siliconflow.com/v1",
)
register_provider(
    "gemini",
    api_key_env="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
register_provider("local", is_local=True)

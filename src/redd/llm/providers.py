from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, TypeVar, cast

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from redd.exceptions import RuntimeDependencyError

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ProviderSpec:
    """Runtime information for a model provider."""

    canonical_name: str
    api_key_env: str | None = None
    base_url: str | None = None
    litellm_provider: str | None = None
    is_local: bool = False


@dataclass(frozen=True)
class LLMConfig:
    """Provider-agnostic LLM execution config."""

    mode: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 5
    wait_time: float = 10.0
    structured_backend: str = "auto"
    provider_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionRequest:
    messages: list[dict[str, Any]]
    response_format: str | None = "json_object"
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str
    raw_response: Any | None = None
    usage: Any | None = None
    model: str | None = None


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
            raise RuntimeDependencyError("Local provider support currently requires CUDA.")

        model_name = str(config.get("llm_model") or config.get("model") or "").strip()
        model_path = str(config.get("llm_model_path") or "").strip()
        if not model_name and not model_path:
            raise ValueError("Local provider requires `llm_model` or `llm_model_path` in config.")

        model_ref = model_path or model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        model_factory = cast(Any, AutoModelForCausalLM)
        self._model = model_factory.from_pretrained(
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
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=None,
            model=None,
        )


def register_provider(
    name: str,
    *,
    api_key_env: str | None = None,
    base_url: str | None = None,
    litellm_provider: str | None = None,
    aliases: tuple[str, ...] = (),
    is_local: bool = False,
) -> None:
    canonical = name.strip().lower()
    spec = ProviderSpec(
        canonical_name=canonical,
        api_key_env=api_key_env,
        base_url=base_url,
        litellm_provider=litellm_provider,
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


def _litellm_model_name(spec: ProviderSpec, model: str) -> str:
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
    provider = spec.litellm_provider
    if not provider:
        return model
    return f"{provider}/{model}"


def build_llm_config(
    mode: str,
    model: str,
    *,
    config: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    **overrides: Any,
) -> LLMConfig:
    spec = get_provider_spec(mode)
    resolved_api_key = get_api_key(config, spec.canonical_name, api_key)
    return LLMConfig(
        mode=spec.canonical_name,
        model=model,
        api_key=resolved_api_key,
        base_url=str(overrides.get("base_url") or (config or {}).get("base_url") or spec.base_url or "")
        or None,
        max_retries=int(overrides.get("max_retries", (config or {}).get("max_retries", 5))),
        wait_time=float(overrides.get("wait_time", (config or {}).get("wait_time", 10.0))),
        structured_backend=str(
            overrides.get("structured_backend", (config or {}).get("structured_backend", "auto"))
        ),
        provider_kwargs=dict(overrides.get("provider_kwargs") or {}),
    )


def _message_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        pass
    if isinstance(response, Mapping):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content") or "")
    return ""


def _response_usage(response: Any) -> Any | None:
    if hasattr(response, "usage"):
        return response.usage
    if isinstance(response, Mapping):
        return response.get("usage")
    return None


def _response_model(response: Any) -> str | None:
    if hasattr(response, "model"):
        return response.model
    if isinstance(response, Mapping):
        model = response.get("model")
        return str(model) if model is not None else None
    return None


class LLMRuntime:
    """Shared LLM execution facade for text and typed completions."""

    def __init__(self, config: LLMConfig, *, local_config: Mapping[str, Any] | None = None) -> None:
        self.config = config
        self.spec = get_provider_spec(config.mode)
        self._local_client = LocalChatClient(local_config or {}) if self.spec.is_local else None
        self._instructor_client: Any | None = None

    @classmethod
    def from_config(
        cls,
        mode: str,
        model: str,
        *,
        config: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        **overrides: Any,
    ) -> "LLMRuntime":
        llm_config = build_llm_config(
            mode,
            model,
            config=config,
            api_key=api_key,
            **overrides,
        )
        return cls(llm_config, local_config=config)

    @property
    def litellm_model(self) -> str:
        return _litellm_model_name(self.spec, self.config.model)

    def _completion_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.litellm_model,
            "messages": request.messages,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url
        if request.response_format and request.response_format != "text":
            kwargs["response_format"] = {"type": request.response_format}
        for key in ("temperature", "top_p", "max_tokens", "seed"):
            value = getattr(request, key)
            if value is not None:
                kwargs[key] = value
        kwargs.update(self.config.provider_kwargs)
        return kwargs

    def complete_text(self, request: CompletionRequest) -> CompletionResult:
        if self.spec.is_local:
            if self._local_client is None:
                raise ValueError("Local LLM runtime is not initialized.")
            completion = self._local_client.chat.completions.create(
                model=self.config.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
            )
            return CompletionResult(
                text=_message_content(completion),
                raw_response=completion,
                usage=_response_usage(completion),
                model=_response_model(completion),
            )

        try:
            from litellm import completion
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError("LLM calls require `litellm`.") from exc

        @retry(
            stop=stop_after_attempt(self.config.max_retries + 1),
            wait=wait_exponential(multiplier=self.config.wait_time, min=self.config.wait_time),
            reraise=True,
        )
        def _call() -> Any:
            return completion(**self._completion_kwargs(request))

        response = _call()
        return CompletionResult(
            text=_message_content(response),
            raw_response=response,
            usage=_response_usage(response),
            model=_response_model(response),
        )

    def complete_model(self, request: CompletionRequest, response_model: type[ModelT]) -> ModelT:
        if self.config.structured_backend in {"auto", "instructor"} and not self.spec.is_local:
            try:
                return self._complete_model_with_instructor(request, response_model)
            except ModuleNotFoundError:
                if self.config.structured_backend == "instructor":
                    raise

        result = self.complete_text(
            CompletionRequest(
                messages=request.messages,
                response_format="json_object",
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                seed=request.seed,
            )
        )
        return response_model.model_validate(json.loads(result.text))

    def _complete_model_with_instructor(
        self,
        request: CompletionRequest,
        response_model: type[ModelT],
    ) -> ModelT:
        try:
            import instructor
        except ModuleNotFoundError:
            raise

        if self._instructor_client is None:
            self._instructor_client = instructor.from_provider(f"litellm/{self.litellm_model}")
        instructor_client = self._instructor_client

        kwargs: dict[str, Any] = {
            "messages": request.messages,
            "response_model": response_model,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url
        for key in ("temperature", "top_p", "max_tokens", "seed"):
            value = getattr(request, key)
            if value is not None:
                kwargs[key] = value

        @retry(
            stop=stop_after_attempt(self.config.max_retries + 1),
            wait=wait_exponential(multiplier=self.config.wait_time, min=self.config.wait_time),
            reraise=True,
        )
        def _call() -> ModelT:
            return instructor_client.create(**kwargs)

        return _call()


def create_client(
    mode: str,
    *,
    config: Mapping[str, Any] | None = None,
    api_key: str | None = None,
) -> LLMRuntime:
    """Backward-compatible factory returning the shared runtime."""
    model = str((config or {}).get("llm_model") or (config or {}).get("model") or "")
    if not model:
        raise ValueError("LLM client creation requires `llm_model` in config.")
    return LLMRuntime.from_config(mode, model, config=config, api_key=api_key)


def llm_completion(
    mode: str,
    client: Any | None,
    messages: list[dict[str, Any]],
    model: str,
    **kwargs: Any,
) -> str:
    """Backward-compatible text completion wrapper around `LLMRuntime`."""
    runtime = client if isinstance(client, LLMRuntime) else None
    if runtime is None:
        runtime = LLMRuntime.from_config(mode, model, config={"llm_model": model, **kwargs})
    request = CompletionRequest(
        messages=messages,
        response_format=kwargs.get("response_format", "json_object"),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        max_tokens=kwargs.get("max_tokens"),
        seed=kwargs.get("seed"),
    )
    return runtime.complete_text(request).text


register_provider("openai", api_key_env="OPENAI_API_KEY", litellm_provider="openai")
register_provider("deepseek", api_key_env="DEEPSEEK_API_KEY", litellm_provider="deepseek")
register_provider("together", api_key_env="TOGETHER_API_KEY", litellm_provider="together_ai")
register_provider(
    "siliconflow",
    api_key_env="SILICONFLOW_API_KEY",
    base_url="https://api.siliconflow.com/v1",
    litellm_provider="openai",
)
register_provider("gemini", api_key_env="GEMINI_API_KEY", litellm_provider="gemini")
register_provider("local", is_local=True)

"""Shared LLM provider/runtime utilities."""

from __future__ import annotations

from importlib import import_module

from .providers import (
    CompletionRequest,
    CompletionResult,
    LLMConfig,
    LLMRuntime,
    ProviderSpec,
    build_llm_config,
    create_client,
    get_api_key,
    get_provider_spec,
    is_local_provider,
    llm_completion,
    normalize_provider_name,
    register_provider,
)

_HIDDEN_STATE_EXPORTS = {
    "HiddenStatesManager",
    "compare_values",
    "normalize_text",
    "pool_hidden_states",
}


def __getattr__(name: str):
    if name in _HIDDEN_STATE_EXPORTS:
        module = import_module(".hidden_states", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HiddenStatesManager",
    "CompletionRequest",
    "CompletionResult",
    "LLMConfig",
    "LLMRuntime",
    "ProviderSpec",
    "build_llm_config",
    "compare_values",
    "create_client",
    "get_api_key",
    "get_provider_spec",
    "is_local_provider",
    "llm_completion",
    "normalize_text",
    "normalize_provider_name",
    "pool_hidden_states",
    "register_provider",
]

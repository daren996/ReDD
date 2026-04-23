"""Shared LLM provider and prompt utilities."""

from .providers import (
    ProviderSpec,
    create_client,
    get_api_key,
    get_provider_spec,
    is_local_provider,
    llm_completion,
    normalize_provider_name,
    register_provider,
)
from .hidden_states import HiddenStatesManager, compare_values, normalize_text, pool_hidden_states

__all__ = [
    "HiddenStatesManager",
    "ProviderSpec",
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

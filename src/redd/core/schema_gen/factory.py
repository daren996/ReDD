from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from redd.core.llm import is_local_provider, normalize_provider_name


def create_schema_generator(config: Mapping[str, Any], api_key: str | None = None):
    """Create the canonical schema-generation orchestrator."""

    normalized_mode = normalize_provider_name(config["mode"])
    if is_local_provider(normalized_mode):
        raise ValueError("Local schema generation is not implemented in this repository yet.")

    resolved_config = dict(config)
    resolved_config["mode"] = normalized_mode

    unified_module = import_module(".schemagen", __package__)
    generator_cls = getattr(unified_module, "SchemaGen")
    return generator_cls(resolved_config, api_key=api_key)


__all__ = ["create_schema_generator"]

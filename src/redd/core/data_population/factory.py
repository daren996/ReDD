from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from redd.core.llm import normalize_provider_name


def create_data_populator(config: Mapping[str, Any], api_key: str | None = None):
    """Create the canonical data-extraction orchestrator."""

    normalized_mode = normalize_provider_name(config["mode"])
    resolved_config = dict(config)
    resolved_config["mode"] = normalized_mode

    unified_module = import_module(".datapop", __package__)
    populator_cls = getattr(unified_module, "DataPop")
    return populator_cls(resolved_config, api_key=api_key)


__all__ = ["create_data_populator"]

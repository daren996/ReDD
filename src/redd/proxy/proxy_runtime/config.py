"""Internal helpers for proxy-runtime configuration normalization."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def normalize_proxy_runtime_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return normalized proxy runtime configuration."""
    proxy_cfg = config.get("proxy_runtime")
    if isinstance(proxy_cfg, dict):
        return dict(proxy_cfg)
    return {}


def is_proxy_runtime_enabled(config: Mapping[str, Any]) -> bool:
    """Resolve proxy-runtime enablement from the normalized config."""
    proxy_cfg = normalize_proxy_runtime_config(config)
    enabled = proxy_cfg.get("enabled", config.get("use_proxy_runtime", False))
    return bool(enabled)


def resolve_proxy_flag(
    proxy_cfg: Mapping[str, Any],
    primary_key: str,
    default: bool,
) -> bool:
    """Read a proxy flag from normalized proxy configuration."""
    return bool(proxy_cfg.get(primary_key, default))


def resolve_proxy_threshold(
    proxy_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    """Read the configured proxy threshold."""
    return float(proxy_cfg.get("proxy_threshold", config.get("proxy_threshold", 0.5)))


__all__ = [
    "is_proxy_runtime_enabled",
    "normalize_proxy_runtime_config",
    "resolve_proxy_flag",
    "resolve_proxy_threshold",
]

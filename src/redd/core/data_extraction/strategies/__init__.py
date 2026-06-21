"""Internal optimization strategies for unified data extraction."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "AlphaAllocationStrategy": ".alpha_allocation",
    "DocFilteringStrategy": ".doc_filtering",
    "ProxyRuntimeExtractionStrategy": ".proxy_runtime",
}

__all__ = [
    "AlphaAllocationStrategy",
    "DocFilteringStrategy",
    "ProxyRuntimeExtractionStrategy",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

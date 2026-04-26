"""Proxy-related runtime modules."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ConformalProxy",
    "DataExtractionOracle",
    "EmbeddingProxy",
    "GoldenOracle",
    "JoinResolver",
    "PipelineResults",
    "PredicateProxyFactory",
    "PredicateProxyFilter",
    "ProxyDecision",
    "ProxyExecutor",
    "ProxyPipeline",
    "ProxyPipelineConfig",
    "ProxyRuntimeConfig",
    "compute_table_processing_order",
    "create_join_resolver",
    "get_join_graph",
    "parse_join_conditions",
]

_EXPORT_MAP = {
    "ConformalProxy": ".predicate_proxy",
    "DataExtractionOracle": ".proxy_runtime",
    "EmbeddingProxy": ".predicate_proxy",
    "GoldenOracle": ".proxy_runtime",
    "JoinResolver": ".join_resolution",
    "PipelineResults": ".proxy_runtime",
    "PredicateProxyFactory": ".predicate_proxy",
    "PredicateProxyFilter": ".predicate_proxy",
    "ProxyDecision": ".proxy_runtime",
    "ProxyExecutor": ".proxy_runtime",
    "ProxyPipeline": ".proxy_runtime",
    "ProxyPipelineConfig": ".proxy_runtime",
    "ProxyRuntimeConfig": ".proxy_runtime",
    "compute_table_processing_order": ".join_resolution",
    "create_join_resolver": ".join_resolution",
    "get_join_graph": ".join_resolution",
    "parse_join_conditions": ".join_resolution",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)

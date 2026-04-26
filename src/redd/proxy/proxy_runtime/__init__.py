from importlib import import_module

__all__ = [
    "ConformalProxy",
    "DataExtractionOracle",
    "DocumentBatch",
    "EmbeddingProxy",
    "ExecutionStats",
    "GoldenOracle",
    "HardNegative",
    "PipelineResults",
    "ProxyDecision",
    "ProxyExecutor",
    "ProxyPipeline",
    "ProxyPipelineConfig",
    "ProxyRuntimeConfig",
]

_EXPORT_MAP = {
    "ConformalProxy": ".executor",
    "DataExtractionOracle": ".oracle",
    "DocumentBatch": ".executor",
    "EmbeddingProxy": ".executor",
    "ExecutionStats": ".executor",
    "GoldenOracle": ".oracle",
    "HardNegative": ".executor",
    "PipelineResults": ".pipeline",
    "ProxyDecision": ".executor",
    "ProxyExecutor": ".executor",
    "ProxyPipeline": ".pipeline",
    "ProxyPipelineConfig": ".types",
    "ProxyRuntimeConfig": ".executor",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)

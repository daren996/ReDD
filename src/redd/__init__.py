"""Public package surface for ReDD.

External users should import from `redd` directly.
Implementation packages such as `redd.core` and `redd.cli` are internal-only.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "DATA_EXTRACTION": ".api",
    "PREPROCESSING": ".api",
    "SCHEMA_REFINEMENT": ".api",
    "DataLoaderBase": ".api",
    "DataExtractor": ".api",
    "PipelineStage": ".api",
    "SchemaGenerator": ".api",
    "AlphaAllocationConfig": ".parameter_optimization",
    "AlphaAllocationResult": ".parameter_optimization",
    "ConformalProxy": ".predicate_proxy",
    "EmbeddingProxy": ".predicate_proxy",
    "PredicateProxyFactory": ".predicate_proxy",
    "ConformalCalibrationResult": ".parameter_optimization",
    "ConformalCalibrator": ".parameter_optimization",
    "GreedyAlphaAllocator": ".parameter_optimization",
    "PipelineResults": ".proxy_runtime",
    "ProxyDecision": ".proxy_runtime",
    "ProxyExecutor": ".proxy_runtime",
    "ProxyPipeline": ".proxy_runtime",
    "ProxyPipelineConfig": ".proxy_runtime",
    "ProxyRuntimeConfig": ".proxy_runtime",
    "QueryDocumentFilter": ".schema_refine",
    "SchemaTailor": ".schema_refine",
    "StaticSQLAdapter": ".text_to_sql",
    "TextToSQLAdapter": ".text_to_sql",
    "TextToSQLRequest": ".text_to_sql",
    "create_doc_filter": ".doc_filtering",
    "create_join_resolver": ".join_resolution",
    "run_web_demo": ".web_demo",
}


def data_extraction(*args, **kwargs):
    from .data_extraction import data_extraction as _data_extraction

    return _data_extraction(*args, **kwargs)


def create_data_loader(*args, **kwargs):
    from .loader import create_data_loader as _create_data_loader

    return _create_data_loader(*args, **kwargs)


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def preprocessing(*args, **kwargs):
    from .preprocessing import preprocessing as _preprocessing

    return _preprocessing(*args, **kwargs)


def schema_global(*args, **kwargs):
    from .schema_global import schema_global as _schema_global

    return _schema_global(*args, **kwargs)


def schema_refine(*args, **kwargs):
    from .schema_refine import schema_refine as _schema_refine

    return _schema_refine(*args, **kwargs)


def schema_refinement(*args, **kwargs):
    from .schema_refine import schema_refinement as _schema_refinement

    return _schema_refinement(*args, **kwargs)


def run_experiment(*args, **kwargs):
    from .runners import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)


def run_extract(*args, **kwargs):
    from .runners import run_extract as _run_extract

    return _run_extract(*args, **kwargs)


def run_evaluation(*args, **kwargs):
    """Wrapper for experiment-side evaluation workflows."""
    from .runners import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)


def run_preprocessing(*args, **kwargs):
    from .runners import run_preprocessing as _run_preprocessing

    return _run_preprocessing(*args, **kwargs)


def run_schema_refinement(*args, **kwargs):
    from .runners import run_schema_refinement as _run_schema_refinement

    return _run_schema_refinement(*args, **kwargs)


def select_runtime(*args, **kwargs):
    from .experiment import select_runtime as _select_runtime

    return _select_runtime(*args, **kwargs)


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)


__all__ = [
    "DATA_EXTRACTION",
    "PREPROCESSING",
    "SCHEMA_REFINEMENT",
    "DataLoaderBase",
    "DataExtractor",
    "PipelineStage",
    "SchemaGenerator",
    "AlphaAllocationConfig",
    "AlphaAllocationResult",
    "ConformalProxy",
    "EmbeddingProxy",
    "PipelineResults",
    "PredicateProxyFactory",
    "ProxyDecision",
    "ProxyExecutor",
    "ProxyPipeline",
    "ProxyPipelineConfig",
    "ProxyRuntimeConfig",
    "ConformalCalibrationResult",
    "ConformalCalibrator",
    "GreedyAlphaAllocator",
    "QueryDocumentFilter",
    "SchemaTailor",
    "StaticSQLAdapter",
    "TextToSQLAdapter",
    "TextToSQLRequest",
    "create_doc_filter",
    "data_extraction",
    "create_join_resolver",
    "preprocessing",
    "schema_global",
    "schema_refine",
    "schema_refinement",
    "create_data_loader",
    "run_evaluation",
    "run_experiment",
    "run_extract",
    "run_pipeline",
    "run_preprocessing",
    "run_schema_refinement",
    "run_web_demo",
    "select_runtime",
]

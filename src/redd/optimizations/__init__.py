"""Optional optimization modules."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AdaptiveSampler",
    "AdaptiveSamplingMixin",
    "AlphaAllocationConfig",
    "AlphaAllocationResult",
    "AllocationTraceStep",
    "CompositeFilter",
    "ConformalCalibrationResult",
    "ConformalCalibrator",
    "DDGTDocumentSelector",
    "DDGTSampler",
    "DocFilterBase",
    "DocumentSelector",
    "FilterResult",
    "GreedyAlphaAllocator",
    "NoOpFilter",
    "STAGE_DOC_FILTERING",
    "STAGE_PREDICATE_PROXY",
    "SchemaEntropyCalculator",
    "SchemaRelevanceFilter",
    "create_composite_filter",
    "create_doc_filter",
]

_EXPORT_MAP = {
    "AdaptiveSampler": ".adaptive_sampling",
    "AdaptiveSamplingMixin": ".adaptive_sampling",
    "AlphaAllocationConfig": ".alpha_allocation",
    "AlphaAllocationResult": ".alpha_allocation",
    "AllocationTraceStep": ".alpha_allocation",
    "CompositeFilter": ".doc_filtering",
    "ConformalCalibrationResult": ".alpha_allocation",
    "ConformalCalibrator": ".alpha_allocation",
    "DDGTDocumentSelector": ".adaptive_sampling",
    "DDGTSampler": ".adaptive_sampling",
    "DocFilterBase": ".doc_filtering",
    "DocumentSelector": ".adaptive_sampling",
    "FilterResult": ".doc_filtering",
    "GreedyAlphaAllocator": ".alpha_allocation",
    "NoOpFilter": ".doc_filtering",
    "STAGE_DOC_FILTERING": ".alpha_allocation",
    "STAGE_PREDICATE_PROXY": ".alpha_allocation",
    "SchemaEntropyCalculator": ".adaptive_sampling",
    "SchemaRelevanceFilter": ".doc_filtering",
    "create_composite_filter": ".doc_filtering",
    "create_doc_filter": ".doc_filtering",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)

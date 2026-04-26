"""Alpha allocation package."""

from redd.core.utils.conformal_calibration import ConformalCalibrationResult, ConformalCalibrator

from .data_extraction_adapter import DataExtractionAlphaAllocator
from .greedy_allocator import GreedyAlphaAllocator
from .types import (
    STAGE_DOC_FILTERING,
    STAGE_PREDICATE_PROXY,
    AllocationTraceStep,
    AlphaAllocationConfig,
    AlphaAllocationResult,
)

__all__ = [
    "AlphaAllocationConfig",
    "AlphaAllocationResult",
    "AllocationTraceStep",
    "ConformalCalibrationResult",
    "ConformalCalibrator",
    "DataExtractionAlphaAllocator",
    "GreedyAlphaAllocator",
    "STAGE_DOC_FILTERING",
    "STAGE_PREDICATE_PROXY",
]

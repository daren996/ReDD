"""Alpha allocation package."""

from redd.core.utils.conformal_calibration import ConformalCalibrationResult, ConformalCalibrator

from .datapop_adapter import DataPopAlphaAllocator
from .greedy_allocator import GreedyAlphaAllocator
from .types import (
    AlphaAllocationConfig,
    AlphaAllocationResult,
    AllocationTraceStep,
    STAGE_DOC_FILTERING,
    STAGE_PREDICATE_PROXY,
)

__all__ = [
    "AlphaAllocationConfig",
    "AlphaAllocationResult",
    "AllocationTraceStep",
    "ConformalCalibrationResult",
    "ConformalCalibrator",
    "DataPopAlphaAllocator",
    "GreedyAlphaAllocator",
    "STAGE_DOC_FILTERING",
    "STAGE_PREDICATE_PROXY",
]

"""
Adaptive Sampling Module for Schema Generation.

This module provides adaptive sampling strategies to reduce the number of documents
processed while maintaining schema quality guarantees.

Strategies:
- Entropy-based: Uses schema entropy and stability streaks
- DDGT: Uses diversity-driven sampling with Good-Turing stopping condition
"""

from .ddgt.document_selector import DDGTDocumentSelector
from .ddgt.sampler import DDGTSampler
from .entropy.document_selector import DocumentSelector
from .entropy.sampler import AdaptiveSampler
from .mixin import AdaptiveSamplingMixin
from .schema_entropy import SchemaEntropyCalculator

__all__ = [
    "AdaptiveSamplingMixin",
    "SchemaEntropyCalculator",
    "AdaptiveSampler",
    "DocumentSelector",
    "DDGTSampler",
    "DDGTDocumentSelector",
]

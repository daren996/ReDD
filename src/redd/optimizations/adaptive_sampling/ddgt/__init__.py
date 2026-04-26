"""
DDGT (Diversity-Driven Good-Turing) Adaptive Sampling.

This submodule implements the DDGT adaptive sampling strategy
that uses diversity-driven document selection and Good-Turing stopping
condition for probabilistic coverage guarantees.
"""

from .document_selector import DDGTDocumentSelector
from .sampler import DDGTSampler

__all__ = [
    "DDGTSampler",
    "DDGTDocumentSelector",
]
